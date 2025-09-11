from pathlib import Path
import pickle
import time

import lightning.pytorch as pl
from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch import optim
from tqdm import tqdm
from sklearn.manifold import TSNE
import wandb

from networks.imaging_decoders import ImagingMaskedDecoder
from networks.imaging_encoders import ImagingMaskedEncoder
from networks.tokenizers import *
from utils.logging_related import imgs_to_wandb_video, CustomWandbLogger
from utils.losses import ReconstructionCriterion, calculate_psnr
from utils.imaging_model_related import patchify, unpatchify


class BasicModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.train_epoch_start_time = None
        self.val_epoch_start_time = None
        self.test_epoch_start_time = None
        self.patchify_method = None
        self.unpatchify_method = None
        self.module_logger = CustomWandbLogger()

        self.test_psnr = []
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.time()
        self.module_logger.reset_item()
        
    def on_test_epoch_start(self) -> None:
        self.test_epoch_start_time = time.time()
        self.module_logger.reset_test_item()

    def on_validation_epoch_end(self) -> None:
        self.wandb_log(self.current_epoch, mode="val")

    def on_train_epoch_end(self) -> None:
        epoch_runtime = (time.time() - self.train_epoch_start_time) / 60
        self.module_logger.update_metric_item("train/epoch_runtime", epoch_runtime, mode="train")
        self.module_logger.update_metric_item("train/lr", self.hparams.lr, mode="train")
        self.wandb_log(self.current_epoch, mode="train")
    
    def on_test_epoch_end(self) -> None:
        epoch_runtime = (time.time() - self.test_epoch_start_time) / 60
        self.module_logger.update_metric_item("test/epoch_runtime", epoch_runtime, mode="test")
        self.wandb_log(self.current_epoch, mode="test")
    
    def log_recon_metrics(self, loss_dict, psnr_value, mode="train"):
        for loss_name, loss_value in loss_dict.items():
            self.module_logger.update_metric_item(f"{mode}/recon_{loss_name}", loss_value.detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/recon_psnr", psnr_value, mode=mode)
    
    def log_recon_videos(self, pred_patches, gt_imgs, sub_idx, mode="train"):
        sub_path = eval(f"self.trainer.datamodule.{mode}_dset").subject_paths[sub_idx]
        sub_id = sub_path.parent.name
        # Extend batch dimension for calculation
        pred_patches = pred_patches[None]
        gt_imgs = gt_imgs[None]
        # Prepare for wandb video logging: (B, S, T, H, W) -> (T, C, H, W)
        pred_imgs = unpatchify(pred_patches, im_shape=gt_imgs.shape, patch_size=self.hparams.patch_size)
        gt_pred_img_log = imgs_to_wandb_video(torch.cat([gt_imgs[0], pred_imgs[0]], dim=2))
        self.module_logger.update_video_item(f"{mode}_video/pred_imgs", sub_id, gt_pred_img_log, mode=mode)
    
    def wandb_log(self, epoch=0, mode="train"):
        # Log images and videos
        for item in eval(f"self.module_logger.log_{mode}_img_dict"):
            img_list = list(eval(f"self.module_logger.log_{mode}_img_dict)")[item].values())
            if img_list:
                self.logger.experiment.log({item: img_list}, commit=False)
        for item in eval(f"self.module_logger.log_{mode}_video_dict"):
            video_list = list(eval(f"self.module_logger.log_{mode}_video_dict")[item].values())
            if video_list:
                self.logger.experiment.log({item: video_list}, commit=False)

        # Log metrics
        for item in eval(f"self.module_logger.log_{mode}_metric_dict"):
            value_list = eval(f"self.module_logger.log_{mode}_metric_dict")[item]
            epoch_avg = sum(value_list) / len(value_list)
            eval(f"self.module_logger.log_{mode}_metric_dict")[item] = epoch_avg
        self.log_dict(eval(f"self.module_logger.log_{mode}_metric_dict"), on_epoch=True, on_step=False)
        
        # Upload all logging data online with commit flag be set to True
        if mode in ["train", "test"]:
            self.log("epoch", epoch, on_epoch=True, on_step=False)
        
    def wandb_log_final(self):
        test_table = wandb.Table(data=self.wandb_infer.data_list, columns=self.wandb_infer.save_table_column)
        self.log({"test_table": test_table}, on_epoch=True, on_step=False)


class ReconMAE(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.data_view = self.hparams.val_dset.view
        use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False # For positional embedding
        img_shape = self.hparams.val_dset[0][0].shape

        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, use_both_axes=use_both_axes, **self.hparams)
        self.decoder_imaging = ImagingMaskedDecoder(num_patches=self.encoder_imaging.patch_embed.num_patches, 
                                                    grid_size=self.encoder_imaging.patch_embed.grid_size,
                                                    use_both_axes=use_both_axes,
                                                    head_out_dim=self.encoder_imaging.patch_p_num,
                                                    **self.hparams)
        self.reconstruction_criterion = ReconstructionCriterion(**kwargs)
        
    def forward(self, x):
        x, mask, ids_restore = self.encoder_imaging(x)
        x = self.decoder_imaging(x, ids_restore)
        return x, mask
        
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, sub_idx = batch

        pred_patches, mask = self.forward(imgs)

        imgs_patches = patchify(imgs, patch_size=self.hparams.patch_size)
        loss_dict = self.reconstruction_criterion(pred_patches, imgs_patches, mask) 
        psnr_value = calculate_psnr(pred_patches, imgs_patches, mask, replace_with_gt=True)
        
        # Logging metrics and median
        self.log_recon_metrics(loss_dict, psnr_value, mode=mode)

        if mode == "val": # For checkpoint tracking
            self.module_logger.update_metric_item(f"{mode}_PSNR", psnr_value, mode=mode) 
            
        log_rate = eval(f"self.hparams.{mode}_log_rate")
        if self.current_epoch > 0 and ((self.current_epoch + 1) % log_rate == 0):
            if (sub_idx == 0).any():
                i = (sub_idx == 0).argwhere().squeeze().item()
                self.log_recon_videos(pred_patches[i], imgs[i], sub_idx[i], mode=mode)
            if (sub_idx == 1).any():
                i = (sub_idx == 1).argwhere().squeeze().item()
                self.log_recon_videos(pred_patches[i], imgs[i], sub_idx[i], mode=mode)
        return loss_dict["loss"]
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, sub_idx = batch

        pred_patches, mask = self.forward(imgs)

        pred_imgs = unpatchify(pred_patches, im_shape=imgs.shape, patch_size=self.hparams.patch_size)
        psnr_value = calculate_psnr(pred_imgs, imgs, reduction="none") # (B, S)
        psnr_value = np.mean(psnr_value, axis=2)
        self.test_psnr.append(psnr_value)

        # Save sample images
        if (sub_idx == 0).any():
            i = (sub_idx == 0).argwhere().squeeze().item()
            sample_imgs = imgs[i]
            sample_recon_imgs = pred_imgs[i]
            self.log_recon_videos(pred_patches[i], imgs[i], sub_idx[i], mode="test")

            if self.hparams.test_sample_path is not None:
                Path(self.hparams.test_sample_path).mkdir(parents=True, exist_ok=True)
                view_name = self.hparams.test_sample_path.split('_')[-1]
                sample_imgs = sample_imgs.detach().cpu().numpy()
                sample_recon_imgs = sample_recon_imgs.detach().cpu().numpy()
                for k in range(sample_imgs.shape[0]):
                    gt = sample_imgs[k, 13]
                    recon = sample_recon_imgs[k, 13]
                    plt.imsave(Path(self.hparams.test_sample_path) / f"gt_0_s{k}_t13.png", gt, cmap="gray")
                    plt.imsave(Path(self.hparams.test_sample_path) / f"recon_{view_name}_0_s{k}_t13.png", recon, cmap="gray")

    def on_test_epoch_end(self) -> None:
        test_psnr = np.concatenate(self.test_psnr, axis=0) # (#test, S)
        psnr_mean = np.mean(test_psnr)
        psnr_std = np.std(test_psnr)
        # Short-axis
        if self.data_view == 0:
            psnr_sax_mean = psnr_mean
            psnr_sax_std = psnr_std
            psnr_lax_mean = 0
            psnr_lax_std = 0
        # Long-axis
        elif self.data_view == 1:
            psnr_sax_mean = 0
            psnr_sax_std = 0
            psnr_lax_mean = psnr_mean
            psnr_lax_std = psnr_std
        elif self.data_view == 2:
            psnr_lax_mean = np.mean(test_psnr[:, :3])
            psnr_lax_std = np.std(test_psnr[:, :3])
            psnr_sax_mean = np.mean(test_psnr[:, 3:])
            psnr_sax_std = np.std(test_psnr[:, 3:])
        results = {"psnr_mean": psnr_mean, "psnr_std": psnr_std, 
                   "psnr_sax_mean": psnr_sax_mean, "psnr_sax_std": psnr_sax_std, 
                   "psnr_lax_mean": psnr_lax_mean, "psnr_lax_std": psnr_lax_std, }
        
        # Table logging
        columns = list(results.keys())
        test_table = wandb.Table(columns=columns)
        test_table.add_data(*[results[col] for col in columns])
        wandb.log({"Evaluation_table": test_table})

        if self.hparams.test_psnr_path is not None:
            Path(self.hparams.test_psnr_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.hparams.test_psnr_path, "wb") as file:
                pickle.dump(results, file)

        self.wandb_log(self.current_epoch, mode="test")
                
    @torch.no_grad()
    def generate_latents(self, data_loader, 
                         token_path: str, 
                         tsne_map_path: str,
                         save_all_patch_tokens: bool = True,
                         save_tsne: bool = True,
                         ):
        latents = []
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Generating latent codes"):
            imgs, sub_idx = batch
            if torch.cuda.is_available():
                imgs = imgs.to("cuda")
            sub_path = data_loader.dataset.subject_paths[sub_idx]
            sub_id = sub_path.parent.name
            enc_output_latent, _, ids_restore = self.encoder_imaging(imgs)
            cls_t = enc_output_latent[:, 0, :]
            all_t_ = enc_output_latent[:, 1:, :]
            
            # Restore the order of the tokens
            all_t_restore = torch.gather(all_t_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, all_t_.shape[-1]))
            
            # Take the average of tokens across spatial dimensions
            B, S, T = imgs.shape[:3]
            num_t_patches = T // self.hparams.patch_size[0]
            all_t_restore = all_t_restore.reshape(B, S, num_t_patches, -1, all_t_restore.shape[-1])
            all_t_restore = all_t_restore.moveaxis(1, 2)
            all_t = all_t_restore.reshape(B, num_t_patches, -1, all_t_restore.shape[-1]).mean(2)
            
            if save_all_patch_tokens:
                latents.append({'subj_id': np.array(int(sub_id)).reshape(1, 1), 
                                'cls_token': cls_t.detach().cpu().numpy(), 
                                'all_token': all_t.detach().cpu().numpy()})
            else:
                latents.append({'subj_id': np.array(int(sub_id)).reshape(1, 1), 
                                'cls_token': cls_t.detach().cpu().numpy(),})
        
        # Save the latent codes
        if not save_all_patch_tokens:
            cls_token = np.concatenate([i['cls_token'] for i in latents])
            subj_id = np.concatenate([i['subj_id'] for i in latents]).reshape(-1)
            np.savez(token_path, cls_token=cls_token, subj_id=subj_id)
        else:
            cls_token = np.concatenate([i['cls_token'] for i in latents])
            subj_id = np.concatenate([i['subj_id'] for i in latents]).reshape(-1)
            all_token = np.concatenate([i['all_token'] for i in latents])
            np.savez(token_path, cls_token=cls_token, all_token=all_token, subj_id=subj_id)
        
        if save_tsne:
        # Save the t-SNE embeddings
            scaled_avg = StandardScaler().fit_transform(np.mean(all_token, axis=1))
            scaled_cls = StandardScaler().fit_transform(cls_token)
            scaled_tmp = StandardScaler().fit_transform(all_token.reshape(-1, all_token.shape[2]))
            
            tsne_map_cls = TSNE(n_components=3, perplexity=5, learning_rate="auto").fit_transform(scaled_cls)
            tsne_map_avg = TSNE(n_components=3, perplexity=5, learning_rate="auto").fit_transform(scaled_avg)
            tsne_map_tmp = TSNE(n_components=3, perplexity=5, learning_rate="auto").fit_transform(scaled_tmp)
                
            np.savez(tsne_map_path, 
                    tsne_map_cls=tsne_map_cls, 
                    tsne_map_avg=tsne_map_avg, 
                    tsne_map_tmp=tsne_map_tmp, 
                    subj_id=subj_id)
