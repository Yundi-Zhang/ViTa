import pickle
import torch
from torch import nn
from timm.models.vision_transformer import Block
from torchvision.models import resnet18, resnet50
import wandb
from models.reconstruction_models import BasicModule
from networks.imaging_decoders import LinearDecoder
from networks.imaging_encoders import ImagingMaskedEncoder
from networks.tokenizers import *
from utils.losses import RegressionCriterion, NumericalReconCriterion
from utils.imaging_model_related import sincos_pos_embed


class RegrMAE(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()
        self.regressor_type = kwargs.get("regressor_type")
        self.use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False # For positional embedding
        img_shape = self.hparams.val_dset[0][0].shape
        self.S_sax = self.hparams.val_dset.sax_slice_num
        self.test_gt = []
        self.test_pred = []
        self.test_subject_id = []
        # --------------------------------------------------------------------------
        # MAE encoder
        self.hparams.mask_ratio = 0.
        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, use_both_axes=self.use_both_axes, 
                                                    S_sax=self.S_sax, **self.hparams)
        # --------------------------------------------------------------------------
        # Regression decoder and head
        self.dec_embed = nn.Linear(self.hparams.enc_embed_dim, self.hparams.dec_embed_dim, bias=True)
        self.dec_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.encoder_imaging.patch_embed.num_patches, self.hparams.dec_embed_dim), requires_grad=False) # with cls token
        if self.regressor_type == "linear":
            self.regressor = LinearDecoder(self.hparams.dec_embed_dim * self.encoder_imaging.patch_embed.num_patches, 
                                           self.hparams.dec_embed_dim, self.hparams.dec_depth)
        elif self.regressor_type == "cls_token":
            self.regressor = nn.AdaptiveAvgPool2d((1, 1))
        self.regression_criterion = RegressionCriterion(**kwargs)
    
    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding
        dec_pos_embed = sincos_pos_embed(self.hparams.dec_embed_dim, self.encoder_imaging.patch_embed.grid_size, cls_token=True,
                                         use_both_axes=self.use_both_axes)
        self.dec_pos_embed.data.copy_(dec_pos_embed.unsqueeze(0))
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
            
    
    def forward_decoder(self, x):
        """Forwrard pass of regression decoder
        input:
            latent: [B, num_patches, embed_dim] torch.Tensor
        output:
            pred: [B, 1] torch.Tensor
        """
        # Embed tokens
        x = self.dec_embed(x)
        
        # Append mask tokens and add positional embedding in schuffled order
        dec_pos_embed = self.dec_pos_embed.repeat(x.shape[0], 1, 1)
        x = x + dec_pos_embed # TODO: Dropput
        if self.regressor_type == "linear":
            x = x[:, 1:, :] # remove cls token
        elif self.regressor_type == "cls_token":
            x = x[:, :1, :] # keep cls token only
        else:
            raise NotImplementedError
        # Regression decoder
        x = self.regressor(x) # apply regressor
        x = x.squeeze(-1) if self.regressor_type == "cls_token" else x
        x = torch.relu(x) #[B, 1]
        return x
    
    def forward(self, imgs):
        latent, _, _ = self.encoder_imaging(imgs)
        x = self.forward_decoder(latent)
        return x
    
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, values, sub_idx = batch
        
        pred_values = self.forward(imgs)
        loss, mae = self.regression_criterion(pred_values, values)
        
        # Logging metrics and median
        self.module_logger.update_metric_item(f"{mode}/regr_loss", loss.detach().item(), mode=mode)
        self.module_logger.update_metric_item(f"{mode}/mae", mae, mode=mode)
        if mode == "val":
            self.log_dict({f"{mode}_MAE": mae}) # For checkpoint tracking # TODO 
        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, values, sub_idx = batch
        
        pred_values = self.forward(imgs)
        loss, mae = self.regression_criterion(pred_values, values)
        self.test_gt.append(values.detach().cpu())
        self.test_pred.append(pred_values.detach().cpu())
        self.test_subject_id.append(sub_idx.detach())
        # Logging metrics and median
        self.module_logger.update_metric_item(f"test/regr_loss", loss.detach().item(), mode="test")
        self.module_logger.update_metric_item(f"test/mae", mae, mode="test")
    
    def on_test_epoch_end(self) -> None:
        gts = torch.concat(self.test_gt, dim=0)
        preds = torch.concat(self.test_pred, dim=0)
        subjects_ids = torch.concat(self.test_subject_id, dim=0)
        results = {"gts": gts, "preds": preds, "subjects_ids": subjects_ids, }
        if self.hparams.test_results_path is not None:
            with open(self.hparams.test_results_path, "wb") as file:
                pickle.dump(results, file)

        # Calculate mae and std and log as a table
        table = wandb.Table(columns=["Feature name", "mae", "std"])
        mask = ~torch.isnan(gts)
        for i in range(gts.shape[1]):
            name = self.hparams.selected_features[i]
            m = mask[:, i]
            masked_gts = gts[:, i][m]
            masked_preds = preds[:, i][m]
            absolute_errors = torch.abs(masked_preds - masked_gts)
            mae = absolute_errors.mean().item()
            std = torch.std(absolute_errors).item()
            table.add_data(name.replace("/", "_"), mae, std)
        wandb.log({"Evaluation_table": table})

        return {"mae":mae, "std":std}


class ResNet18Module(BasicModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        
        val_dataset= kwargs.get("val_dset")
        self.img_shape = val_dataset[0][0].shape
        self.num_classes = val_dataset[0][1].shape[0]
        self.network = self._initial_network()
        # self.regression_criterion = RegressionCriterion(**kwargs)
        self.regression_criterion = NumericalReconCriterion(**kwargs)
        
        self.test_gt = []
        self.test_pred = []
        self.test_subject_id = []
        self.test_results_path = None
    
    def _initial_network(self):
        S, T = self.img_shape[:2]
        _network = resnet18(pretrained=False, num_classes=self.num_classes)
        _network.conv1 = torch.nn.Conv2d(in_channels=S*T, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        return _network
    
    def forward(self, x):
        input_shape = x.shape # [B, S, T, H, W]
        x = x.view(input_shape[0], -1, input_shape[-2], input_shape[-1]) # [B, S*T, H, W]
        x = self.network(x)
        x = torch.relu(x) # [B, 1]
        return x
    
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, values, sub_idx = batch
        
        pred_values = self.forward(imgs)
        losses = self.regression_criterion(pred_values, values)
        loss = losses["loss"]
        # Logging metrics and median
        self.module_logger.update_metric_item(f"{mode}/regr_loss", loss.detach().item(), mode=mode)
        for k,v in losses.items():
            if k == "loss": continue
            else:
                if k.startswith("r2"):
                    self.module_logger.update_metric_item(f"{mode}_r2/{k}", v, mode=mode)
                else:
                    self.module_logger.update_metric_item(f"{mode}/{k}", v, mode=mode)
        if mode == "val":
            self.log_dict({f"{mode}_MAE": losses["mean_mae"]}) # For checkpoint tracking # TODO 
        return loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, values, sub_idx = batch
        pred_values = self.forward(imgs)
        
        self.test_gt.append(values.detach().cpu())
        self.test_pred.append(pred_values.detach().cpu())
        self.test_subject_id.append(sub_idx.detach())
        # Logging metrics and median
    
    def on_test_epoch_end(self) -> None:
        gts = torch.concat(self.test_gt, dim=0)
        preds = torch.concat(self.test_pred, dim=0)
        subjects_ids = torch.concat(self.test_subject_id, dim=0)
        results = {"gts": gts, "preds": preds, "subjects_ids": subjects_ids, }
        if self.hparams.test_results_path is not None:
            with open(self.hparams.test_results_path, "wb") as file:
                pickle.dump(results, file)
        
        # Calculate mae and std and log as a table
        table = wandb.Table(columns=["Feature name", "mae", "std"])
        mask = ~torch.isnan(gts)
        for i in range(gts.shape[1]):
            name = self.hparams.selected_features[i]
            m = mask[:, i]
            masked_gts = gts[:, i][m]
            masked_preds = preds[:, i][m]
            absolute_errors = torch.abs(masked_preds - masked_gts)
            mae = absolute_errors.mean().item()
            std = torch.std(absolute_errors).item()
            table.add_data(name.replace("/", "_"), mae, std)
        wandb.log({"Evaluation_table": table})
        

class ResNet50Module(ResNet18Module):
    
    def _initial_network(self):
        S, T = self.img_shape[:2]
        _network = resnet50(pretrained=False, num_classes=self.num_classes)
        _network.conv1 = torch.nn.Conv2d(in_channels=S*T, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        return _network
