from pathlib import Path
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from timm.models.vision_transformer import Block
from monai.losses import DiceLoss
from tqdm import tqdm
import wandb
from models.reconstruction_models import BasicModule
from networks.imaging_decoders import UNETR_decoder
from networks.imaging_encoders import ImagingMaskedEncoder
from networks.tokenizers import *
from utils.losses import SegmentationCriterion
from utils.general import to_1hot, hausdorff_distance
from utils.logging_related import imgs_to_wandb_video
from utils.imaging_model_related import sincos_pos_embed


class SegMAE(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.save_hyperparameters()
        self.test_dice_scores = []
        self.test_hd95 = []

        self.num_classes = self.hparams.val_dset.num_classes
        self.data_view = self.hparams.val_dset.view
        self.use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False # For positional embedding
        img_shape = self.hparams.val_dset[0][0].shape
        # --------------------------------------------------------------------------
        # MAE encoder
        self.hparams.mask_ratio = 0.
        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, use_both_axes=self.use_both_axes, **self.hparams)
        # --------------------------------------------------------------------------
        # Segmentation decoder and head
        self.decoder_embed = nn.Linear(self.hparams.enc_embed_dim, self.hparams.dec_embed_dim // img_shape[0], bias=True)
        self.decoder = UNETR_decoder(in_channels=img_shape[0],
                                     out_channels=self.num_classes,
                                     img_size=img_shape,
                                     patch_size=self.hparams.patch_size,
                                     feature_size=self.hparams.feature_size,
                                     upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
                                     hidden_size=self.hparams.dec_embed_dim,
                                     spatial_dims=self.hparams.spatial_dims)
        self.segmentation_criterion = SegmentationCriterion(data_view=self.data_view, num_classes=self.num_classes, **kwargs)
    
    def forward_decoder(self, imgs, encoder_output, hidden_latents):
        # Apply UNETR decoder
        decoder_imgs = imgs
        decoder_x = self.decoder_embed(encoder_output)[:, 1:, :]
        decoder_hidden_states = []
        for i in range(len(hidden_latents)):
            if i not in [2, 4]:
                continue
            else:
                decoder_hidden_states.append(self.decoder_embed(hidden_latents[i]))
        decoder_output = self.decoder(x_in=decoder_imgs, x=decoder_x, hidden_states_out=decoder_hidden_states)
        preds = torch.nn.functional.softmax(decoder_output, dim=1)
        return preds
    
    def forward(self, imgs):
        encoder_output, hidden_latents = self.encoder_imaging.forward_with_skip_connection(imgs)
        pred_segs = self.forward_decoder(imgs, encoder_output, hidden_latents)
        return pred_segs
    
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, segs, sub_idx = batch
        
        pred_segs_ = self.forward(imgs)
        gt_segs_ = to_1hot(segs, num_class=self.num_classes) # (B, S, T, H, W, class)
        gt_segs_ = gt_segs_.moveaxis(-1, 1) # (B, class, S, T, H, W)
        loss, dice_score = self.segmentation_criterion(pred_segs_, gt_segs_, mode=mode)
        
        # Logging metrics and median
        self.log_seg_metrics(loss, dice_score, mode=mode)
        if mode == "val":
            self.log_dict({f"{mode}_Dice_FG": dice_score[1:].mean().detach().item()}) # For checkpoint tracking
            
        log_rate = eval(f"self.hparams.{mode}_log_rate")
        if self.current_epoch > 0 and ((self.current_epoch + 1) % log_rate == 0):
            if (sub_idx == 0).any():
                i = (sub_idx == 0).argwhere().squeeze().item()
                pred_seg = torch.argmax(pred_segs_[i], dim=0).detach()
                self.log_seg_videos(imgs[i], segs[i], pred_seg, sub_idx[i], mode=mode)
            if (sub_idx == 1).any():
                i = (sub_idx == 1).argwhere().squeeze().item()
                pred_seg = torch.argmax(pred_segs_[i], dim=0).detach()
                self.log_seg_videos(imgs[i], segs[i], pred_seg, sub_idx[i], mode=mode)

        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    def on_test_epoch_start(self) -> None:
        self.module_logger.reset_test_item()
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, segs, sub_idx = batch
        
        pred_segs_ = self.forward(imgs)
        gt_segs_ = to_1hot(segs, num_class=self.num_classes) # (B, S, T, H, W, class)
        gt_segs_ = gt_segs_.moveaxis(-1, 1) # (B, class, S, T, H, W)

        _, dice_scores = self.segmentation_criterion(pred_segs_, gt_segs_, mode="test")

        pred_segs_ = pred_segs_.moveaxis(1, 2)
        gt_segs_ = gt_segs_.moveaxis(1, 2)
        B, S = pred_segs_.shape[:2]
        slice_mask = torch.ones((B, S), dtype=torch.bool)
        if self.data_view != 0: # If it is either long-axis or both axes
            slice_mask[:, 1] = 0
        pred_segs_ = pred_segs_[slice_mask] # (S, C, T, H, W)
        gt_segs_ = gt_segs_[slice_mask] # (S, C, T, H, W)

        pred_seg = torch.argmax(pred_segs_, dim=1) # (S, T, H, W)
        pred_seg_1hot_ = to_1hot(pred_seg, num_class=self.num_classes) # (S, T, H, W, C)
        pred_seg_1hot_ = pred_seg_1hot_.moveaxis(-1, 1) # (S, C, T, H, W)

        hd = torch.empty(pred_seg_1hot_.shape[:2])
        for i in range(pred_seg_1hot_.shape[0]):
            hd[i] = hausdorff_distance(pred_seg_1hot_[i], gt_segs_[i])
        hd95_mean = hd.mean(dim=0)

        self.test_dice_scores.append(dice_scores)
        self.test_hd95.append(hd95_mean)

        del pred_segs_, gt_segs_, pred_seg_1hot_, hd
        torch.cuda.empty_cache()
        
    def on_test_epoch_end(self) -> None:
        test_dice_scores = torch.stack(self.test_dice_scores, dim=1)
        with open(f"{self.hparams.test_scores_path}", 'wb') as f:
            np.save(f, np.array(test_dice_scores.detach().cpu()))

        test_hd95 = torch.stack(self.test_hd95, dim=1)
        hd95_path = self.hparams.test_scores_path.replace("dice", "hd95")
        with open(hd95_path, 'wb') as f:
            np.save(f, np.array(test_hd95.detach().cpu()))

        table = wandb.Table(columns=["Name", "mae", "std"])
        table.add_data("Dice_FG", test_dice_scores[1:].mean().detach().item(), test_dice_scores[1:].mean(dim=0).std().detach().item())
        table.add_data("Dice_BG", test_dice_scores[0].mean().detach().item(), test_dice_scores[0].std().detach().item())

        if self.data_view == 0 or self.data_view == 2: # For both axes and only short-axis views
            table.add_data("Dice_LVBP", test_dice_scores[1].mean().detach().item(), test_dice_scores[1].std().detach().item())
            table.add_data("Dice_LVMYO", test_dice_scores[2].mean().detach().item(), test_dice_scores[2].std().detach().item())
            table.add_data("Dice_RVBP", test_dice_scores[3].mean().detach().item(), test_dice_scores[3].std().detach().item())
            
        if self.data_view == 1 or self.data_view == 2: # For both axes and only long-axis views
            table.add_data("Dice_LA", test_dice_scores[-2].mean().detach().item(), test_dice_scores[-2].std().detach().item())
            table.add_data("Dice_RA", test_dice_scores[-1].mean().detach().item(), test_dice_scores[-1].std().detach().item())
        wandb.log({"Evaluation_table": table})
        return test_dice_scores
        
    def log_seg_metrics(self, loss, dice_score, mode="train"):
        self.module_logger.update_metric_item(f"{mode}/seg_loss", loss.item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/Dice_FG", dice_score[1:].mean().detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/Dice", dice_score.mean().detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/Dice_BG", dice_score[0].detach().item(), mode=mode)

        if self.data_view == 0 or self.data_view == 2: # For both axes and only short-axis views
            self.module_logger.update_metric_item(f"{mode}/Dice_LVBP", dice_score[1].detach().item(), mode=mode)
            self.module_logger.update_metric_item(f"{mode}/Dice_LVMYO", dice_score[2].detach().item(), mode=mode)
            self.module_logger.update_metric_item(f"{mode}/Dice_RVBP", dice_score[3].detach().item(), mode=mode)

        if self.data_view == 1 or self.data_view == 2: # For both axes and only long-axis views
            self.module_logger.update_metric_item(f"{mode}/Dice_LA", dice_score[-2].detach().item(), mode=mode)
            self.module_logger.update_metric_item(f"{mode}/Dice_RA", dice_score[-1].detach().item(), mode=mode)
    
    def log_seg_videos(self, imgs, segs, pred_segs, sub_idx, mode="train"):
        sub_path = eval(f"self.trainer.datamodule.{mode}_dset").subject_paths[sub_idx]
        sub_id = sub_path.parent.name
        
        # Overlay the segmentations on the groud truth images
        gt_imgs = imgs[:, :, None, ...] # (S, T, H, W)
        gt_imgs = torch.tile(gt_imgs, dims=(1, 1, 3, 1, 1))
        color_pred = torch.zeros_like(gt_imgs)
        color_gt = torch.zeros_like(gt_imgs)
        colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
        for u in torch.unique(segs):
            if u == 0: continue
            color = colors[u]
            for j in range(3):
                color_pred[:, :, j, ...][pred_segs == u] = color[j]
                color_gt[:, :, j, ...][segs == u] = color[j]
        
        pred_overlay = torch.where(color_pred > 0, color_pred, gt_imgs)
        gt_overlay = torch.where(color_gt > 0, color_gt, gt_imgs)
        
        cat_slices = []
        for s in range(gt_imgs.shape[0]):
            if self.data_view != 0 and s == 1: continue
            cat_slices.append(torch.cat([gt_imgs[s], gt_overlay[s], pred_overlay[s]], dim=2))
            
        cat_all = torch.cat([cat_slices[p] for p in range(len(cat_slices))], dim=3) # (T, 3, H_, W_)
        cat_all_log = imgs_to_wandb_video(cat_all, prep=True, in_channel=3)
        self.module_logger.update_video_item(f"{mode}_video/pred_segs", sub_id, cat_all_log, mode=mode)
