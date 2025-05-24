from pathlib import Path
import sys
import time

import lightning.pytorch as pl
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.manifold import TSNE
import wandb

from networks.imaging_decoders import ImagingMaskedDecoder, ViTDecoder
from networks.imaging_encoders import ImagingMaskedEncoder
from networks.tokenizers import *
from utils.logging_related import imgs_to_wandb_video, replace_with_gt_wandb_video, CustomWandbLogger
from utils.losses import ReconstructionCriterion, calculate_psnr
from utils.imaging_model_related import Masker, patchify_SAX, unpatchify_SAX, sincos_pos_embed, patchify, unpatchify
from timm.models.vision_transformer import Block

from utils.train_related import add_weight_decay


class BasicModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.train_epoch_start_time = None
        self.val_epoch_start_time = None
        self.test_epoch_start_time = None
        self.patchify_method = None
        self.unpatchify_method = None
        self.module_logger = CustomWandbLogger()
        
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
    
    def log_recon_videos(self, mask, pred_patches, gt_imgs, sub_idx, mode="train"):
        sub_path = eval(f"self.trainer.datamodule.{mode}_dset").subject_paths[sub_idx]
        sub_id = sub_path.parent.name
        # Extend batch dimension for calculation
        mask = mask[None]
        pred_patches = pred_patches[None]
        gt_imgs = gt_imgs[None]
        # Prepare for wandb video logging: (B, S, T, H, W) -> (T, C, H, W)
        pred_imgs = self.unpatchify_method(pred_patches, im_shape=gt_imgs.shape, patch_size=self.hparams.patch_size, S_sax=self.S_sax, in_channels=self.hparams.patch_in_channels, pixel_unshuffle_scale=self.hparams.pixel_unshuffle_scale)
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
        self.save_hyperparameters() # Save all parameters to self.hparams
        use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False # For positional embedding
        img_shape = self.hparams.val_dset[0][0].shape
        self.S_sax = self.hparams.val_dset.sax_slice_num
        if self.hparams.patch_embed_cls == "PatchEmbed_SAX":
            self.patchify_method = patchify_SAX
            self.unpatchify_method = unpatchify_SAX
        else:
            self.patchify_method = patchify
            self.unpatchify_method = unpatchify
        # self.num_patches = 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hparams.dec_embed_dim), requires_grad=True)
        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, use_both_axes=use_both_axes, 
                                                    S_sax=self.S_sax, **self.hparams)
        self.decoder_imaging = ImagingMaskedDecoder(num_patches=self.encoder_imaging.patch_embed.num_patches, 
                                                    grid_size=self.encoder_imaging.patch_embed.grid_size,
                                                    use_both_axes=use_both_axes,
                                                    **self.hparams)

        self.recon_head = nn.Linear(in_features=self.hparams.dec_embed_dim,
                                    out_features=self.encoder_imaging.patch_p_num,)
        self.reconstruction_criterion = ReconstructionCriterion(**kwargs)
        self.initialize_parameters()
    
    def initialize_parameters(self):        
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights) # TODO
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, mask, ids_restore = self.encoder_imaging(x)
        x = self.decoder_imaging(x, self.mask_token, ids_restore)
        # Reconstruction head
        x = self.recon_head(x)
        x = x[:, 1:, :] # remove cls token
        x = torch.sigmoid(x) # scale x to [0, 1] for reconstruction task
        return x, mask
        
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, sub_idx = batch

        pred_patches, mask = self.forward(imgs)
        imgs_patches = self.patchify_method(imgs, 
                                            patch_size=self.hparams.patch_size, 
                                            S_sax=self.S_sax, 
                                            in_channels=self.hparams.patch_in_channels, 
                                            pixel_unshuffle_scale=self.hparams.pixel_unshuffle_scale)
        # Update the setup of loss calculation to be the same as in Spatiotemporal MAE
        if self.hparams.mask_loss:
            loss_dict = self.reconstruction_criterion(pred_patches, imgs_patches, mask) 
        else:
            loss_dict = self.reconstruction_criterion(pred_patches, imgs_patches)
        psnr_value = calculate_psnr(pred_patches, imgs_patches, mask, replace_with_gt=True)
        
        # Logging metrics and median
        self.log_recon_metrics(loss_dict, psnr_value, mode=mode)
        if mode == "train" or mode == "val":
            if mode == "val": # For checkpoint tracking
                self.module_logger.update_metric_item(f"{mode}_PSNR", psnr_value, mode=mode) 
                
            log_rate = eval(f"self.hparams.{mode}_log_rate")
            if self.current_epoch > 0 and ((self.current_epoch + 1) % log_rate == 0):
                if (sub_idx == 0).any():
                    i = (sub_idx == 0).argwhere().squeeze().item()
                    self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
                if (sub_idx == 1).any():
                    i = (sub_idx == 1).argwhere().squeeze().item()
                    self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
            return loss_dict["loss"]
        
        elif mode == "test": # TODO
            if (sub_idx == 0).any():
                i = (sub_idx == 0).argwhere().squeeze().item()
                self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
            if (sub_idx == 1).any():
                i = (sub_idx == 1).argwhere().squeeze().item()
                self.log_recon_videos(mask[i], pred_patches[i], imgs[i], sub_idx[i], mode=mode)
        
        return loss_dict["loss"]
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="test")
        
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
            if self.hparams.patch_embed_cls.__name__ == "PatchEmbed_SAX":
                S = self.S_sax // self.patch_in_channels + (S - self.S_sax) # 4
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
