import json
import os
from pathlib import Path
import pickle
import time
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import wandb
from termcolor import colored
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from networks.imaging_encoders import ImagingMaskedEncoder
from networks.tabular_decoders import *
from utils.losses import CategoricalReconCriterion, NumericalReconCriterion


class SimCLRTabRecon(pl.LightningModule):
    """
    Lightning module for multimodal SimCLR.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters() # Save all parameters to self.hparams
        
        imaging_hp = self.hparams.imaging_hparams
        tabular_hp = self.hparams.tabular_hparams
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.img_sample = [] # For image logging
        self.val_gt = []
        self.val_pred = []
        self.val_subject_id = []
        self.test_gt = []
        self.test_pred = []
        self.test_logit = []
        self.test_subject_id = []
        self.save_embeddings = None
        self.save_root_path = None
        with open(Path(os.environ["FEATURE_NAMES_OUT"]), 'r') as f:
            self.all_feature_names = json.load(f)
        # --------------------------------------------------------------------------
        # Imaging encoder
        # For positional embedding
        use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False 
        img_shape = self.hparams.val_dset[0][0].shape
        self.S_sax = self.hparams.val_dset.sax_slice_num
        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, 
                                                    use_both_axes=use_both_axes, 
                                                    S_sax=self.S_sax, 
                                                    **imaging_hp.__dict__)
        # --------------------------------------------------------------------------
        # Tabular decoder
        self.decoder_type = tabular_hp.decoder_type
        self.head_type = tabular_hp.decoder_head_type
        self.selected_features = tabular_hp.selected_features
        if self.decoder_type in ["linear", "linear_cls"]:
            if self.decoder_type == "linear":
                tabular_input_dim = self.encoder_imaging.patch_embed.out_channels * self.encoder_imaging.patch_embed.num_patches
            elif self.decoder_type == "linear_cls":
                tabular_input_dim = self.encoder_imaging.patch_embed.out_channels
            self.decoder_tabular = TabularDecoder(input_dim=tabular_input_dim, 
                                                  out_dim=tabular_hp.decoder_dim,
                                                  dim=tabular_hp.decoder_dim, 
                                                  depth=tabular_hp.decoder_depth)
        elif self.decoder_type in ["pool", "pool_cls"]:
            self.decoder_tabular = torch.nn.AdaptiveAvgPool1d(len(self.selected_features))
        # --------------------------------------------------------------------------
        # Head
        if self.head_type == "numerical":
            self.head = NumericalReconHead(input_dim=tabular_hp.decoder_dim, 
                                           output_dim=len(tabular_hp.selected_features))
        elif self.head_type in ["single_categorical"]:
            self.head = SingleCategoricalReconHead(input_dim=tabular_hp.decoder_dim, 
                                                   output_dim=len(tabular_hp.selected_features) * 2)
        elif self.head_type == "multi_categorical":
            self.head = MultipleCategoricalReconHead(input_dim=tabular_hp.decoder_dim,
                                                     selected_features=self.selected_features)
        # --------------------------------------------------------------------------
        # Criterion
        if self.head_type == "numerical":
            self.criterion = NumericalReconCriterion(loss_weights=self.hparams.training_hparams.loss_weights,
                                                     loss_types=self.hparams.training_hparams.loss_types,
                                                     selected_features=self.selected_features,
                                                     use_scalor=tabular_hp.use_scalor)
        else:
            multi_classes = self.head_type == "multi_categorical"
            self.criterion = CategoricalReconCriterion(loss_weights=self.hparams.training_hparams.loss_weights,
                                                       loss_types=self.hparams.training_hparams.loss_types,
                                                       multi_classes=multi_classes,
                                                       selected_features=self.selected_features,)

        # --------------------------------------------------------------------------
        print(colored('Imageing encoder, multimodal:', 'yellow', None, ['bold']), f'{self.encoder_imaging}')
        print(colored('Tabular decoder, multimodal:', 'yellow', None, ['bold']), f'{self.decoder_tabular}\n{self.head}')
        
    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.time()
    
    def on_train_epoch_end(self) -> None:
        epoch_runtime = (time.time() - self.train_epoch_start_time) / 60
        self.log('train/epoch_runtime', epoch_runtime, on_epoch=True, on_step=False)
    
    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        optimizer = torch.optim.Adam(
            [{'params': self.encoder_imaging.parameters()}, 
             {'params': self.decoder_tabular.parameters()},
             {'params': self.head.parameters()}
             ], 
            
            lr=self.hparams.training_hparams.lr, 
            weight_decay=self.hparams.training_hparams.weight_decay,
            )
        
        scheduler = self.initialize_scheduler(optimizer)
        
        return ({"optimizer": optimizer, "lr_scheduler": scheduler})

    def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.hparams.training_hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.training_hparams.dataset_length*self.hparams.training_hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
        elif self.hparams.training_hparams.scheduler == 'anneal':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.training_hparams.warmup_epochs,
                num_training_steps=self.hparams.training_hparams.anneal_max_epochs
            )
        else:
            raise ValueError('Valid schedulers are "cosine" and "anneal"')
        
        return scheduler
    
    def forward(self, imgs):
        all_tokens, _, _ = self.encoder_imaging(imgs)
        if self.decoder_type in ["linear_cls", "pool_cls"]:
            x = all_tokens[:, 0] # class token
        elif self.decoder_type in ["linear", "pool"]:
            x = all_tokens[:, 1:] # all tokens for patches
        x = self.decoder_tabular(x)
        x = self.head(x)
        return x
    
    def training_step(self, batch, batch_idx, mode="train") -> torch.Tensor:
        """
        Trains contrastive model
        """
        imgs, raw_tabs, subject_id = batch
        selected_target = self.get_seleted_target_features(raw_tabs)
        pred = self.forward(imgs) 
        loss_dict = self.criterion(pred, selected_target)
        
        # Logging
        for k, v in loss_dict.items():
            if v is None:
                self.log(f"{mode}/{k}", np.nan, on_epoch=True, on_step=False)
            else:
                self.log(f"{mode}/{k}", v, on_epoch=True, on_step=False)

        return loss_dict["loss"]
            

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, mode="val"):
        imgs, raw_tabs, subject_id = batch
        selected_target = self.get_seleted_target_features(raw_tabs)
        pred = self.forward(imgs) 
        loss_dict = self.criterion(pred, selected_target)
        
        eval(f"self.{mode}_gt").append(selected_target.detach().cpu())
        if self.head_type == "multi_categorical":
            logits = []
            for i in range(len(pred)):
                logit = pred[i].detach().cpu()
                logits.append(logit)
            eval(f"self.{mode}_pred").append(logits)
        else:
            eval(f"self.{mode}_pred").append(pred.detach().cpu())
        eval(f"self.{mode}_subject_id").append(subject_id.detach())
        
        for k, v in loss_dict.items():
            self.log(f"{mode}/{k}", v, on_epoch=True, on_step=False)
            
    @torch.no_grad()
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        _ = self.validation_step(batch, batch_idx, mode="test")
    
    def on_validation_epoch_end(self, mode="val") -> None:
        if self.head_type == "numerical":
            gts = eval(f"torch.concat(self.{mode}_gt, dim=0)")
            subjects_ids = eval(f"torch.concat(self.{mode}_subject_id, dim=0)")
            preds = eval(f"torch.concat(self.{mode}_pred, dim=0)")
            results = {"gts": gts, "preds": preds, "subjects_ids": subjects_ids,}
            self.log_test_numerical_results(gts, preds, mode=mode)

            if mode == "test" and self.hparams.test_results_path is not None:
                with open(self.hparams.test_results_path, "wb") as file:
                    pickle.dump(results, file)
            
        elif self.head_type in ["single_categorical"]:
            gts = eval(f"torch.concat(self.{mode}_gt, dim=0)")
            subjects_ids = eval(f"torch.concat(self.{mode}_subject_id, dim=0)")
            logits = eval(f"torch.concat(self.{mode}_logit, dim=0)")
            self.log_test_single_categorical_results(gts, logits, mode=mode)
            
        elif self.head_type == "multi_categorical":
            gts = eval(f"torch.concat(self.{mode}_gt, dim=0)")
            subjects_ids = eval(f"torch.concat(self.{mode}_subject_id, dim=0)")
            pred_probs, pred_labels = [], []
            for i in range(len(f"self.{mode}_pred"[0])):
                feature_logits = []
                for j in range(len(f"self.{mode}_pred")):
                    feature_logits.append(f"self.{mode}_pred"[j][i])
                feature_logits = torch.concat(feature_logits, dim=0)
                feature_probs = nn.functional.softmax(feature_logits, dim=1)
                feature_labels = torch.argmax(feature_probs, dim=1)
                pred_probs.append(feature_probs)
                pred_labels.append(feature_labels)
            self.log_test_multi_categorical_results(gts, pred_probs, pred_labels, mode=mode)
            
        self.val_gt.clear()
        self.val_pred.clear()
        self.val_subject_id.clear()

    def on_test_epoch_end(self, mode="val") -> None:
        self.on_validation_epoch_end(mode="test")
        
    def log_test_numerical_results(self, gts, preds, mode="test"):
        if mode == "test":
            table = wandb.Table(columns=["Feature name", "mae", "std"])
            
        mask = ~torch.isnan(gts)
        mean_mae = 0.
        for i in range(gts.shape[1]):
            name = self.selected_features[i]
            m = mask[:, i]
            masked_gts = gts[:, i][m]
            masked_preds = preds[:, i][m]
            absolute_errors = torch.abs(masked_preds - masked_gts)
            mae = absolute_errors.mean().item()
            std = torch.std(absolute_errors).item()
            mean_mae += mae
            if mode == "val":
                self.log(f"val/{name}_mae", mae, on_epoch=True, on_step=False)
            if mode == "test":
                table.add_data(name, mae, std)
                
        if mode == "val": # ckpt monitor
            self.log(f"val_mae", mean_mae / gts.shape[1], on_epoch=True, on_step=False)
        if mode == "test":
            wandb.log({"Evaluation_table": table})
    
    def log_test_single_categorical_results(self, gts, logits, mode="test"):
        if mode == "test":
            table = wandb.Table(columns=["Feature name", "accuracy", "f1 score", "auc-roc", "average precision"])
        mean_auc_roc = 0.
        mask = ~torch.isnan(gts)
        for i in range(gts.shape[1]):
            name = self.selected_features[i]
            m = mask[:, i]
            masked_gts = gts[:, i][m].to(torch.int64)
            masked_logits = logits[:, i][m]
            masked_probs = torch.sigmoid(masked_logits)
            masked_labels = torch.argmax(masked_probs, dim=1)
            
            acc = accuracy_score(masked_gts.numpy(), masked_labels.numpy())
            f1 = f1_score(masked_gts.numpy(), masked_labels.numpy(), pos_label=1)
            auc_roc = roc_auc_score(masked_gts.numpy(), masked_probs[:, 1].numpy())
            precision = precision_score(masked_gts.numpy(), masked_probs[:, 1].numpy(), pos_label=1)
            recall = recall_score(masked_gts.numpy(), masked_probs[:, 1].numpy(), pos_label=1)

            mean_auc_roc += auc_roc
            if mode == "val":
                self.log(f"val/{name}_acc", acc, on_epoch=True, on_step=False)
                self.log(f"val/{name}_f1", f1, on_epoch=True, on_step=False)
                self.log(f"val/{name}_auc_roc", auc_roc, on_epoch=True, on_step=False)
                self.log(f"val/{name}_recall", recall, on_epoch=True, on_step=False)
                self.log(f"val/{name}_precision", precision, on_epoch=True, on_step=False)
            if mode == "test":
                table.add_data(name, acc, f1, auc_roc, precision, recall)
        
        if mode == "val": # ckpt monitor
            self.log(f"val_aucroc", mean_auc_roc / gts.shape[1], on_epoch=True, on_step=False)
        if mode == "test":
            wandb.log({"Evaluation_table": table})
        
    def log_test_multi_categorical_results(self, gts, pred_probs, pred_labels, mode="test"):
        if mode == "test":
            table = wandb.Table(columns=["Feature name", "accuracy", "f1 score", "auc-roc"])
        mean_auc_roc = 0.
        mask = ~torch.isnan(gts)
        for i in range(gts.shape[1]):
            num_classes = pred_probs[i].shape[1]
            assert num_classes == list(self.selected_features.values())[i][0]
            
            name = list(self.selected_features.keys())[i]
            m = mask[:, i]
            masked_gts = gts[:, i][m].to(torch.int64).numpy()
            masked_probs = pred_probs[i][m].numpy()
            masked_labels = pred_labels[i][m].numpy()
            
            acc = accuracy_score(masked_gts, masked_labels)
            f1 = f1_score(masked_gts, masked_labels, average="macro")
            try: 
                auc_roc = roc_auc_score(masked_gts, masked_probs, multi_class="ovr", average="macro")
            except:
                auc_roc = float("nan")
            mean_auc_roc += auc_roc
            if mode == "val":
                self.log(f"val/{name}_acc", acc, on_epoch=True, on_step=False)
                self.log(f"val/{name}_f1", f1, on_epoch=True, on_step=False)
                self.log(f"val/{name}_auc_roc", auc_roc, on_epoch=True, on_step=False)
            if mode == "test":
                table.add_data(name, acc, f1, auc_roc,)
                
        if mode == "val": # ckpt monitor
            self.log(f"val_aucroc", mean_auc_roc / gts.shape[1], on_epoch=True, on_step=False)
        if mode == "test":
            wandb.log({"Evaluation_table": table})
        
    def get_seleted_target_features(self, raw_tabs):
        # Get the indices of the selected features
        numerical_num = len(self.all_feature_names["numerical"])
        single_c_num = len(self.all_feature_names["single_categorical"])
        multi_c_num = len(self.all_feature_names["multi_categorical"])
        
        if self.head_type == "numerical":
            all_names = self.all_feature_names["numerical"]
            indices = [all_names.index(name) for name in self.selected_features if name in all_names]
            all_n_target = raw_tabs[:, :numerical_num]
            target = all_n_target[:, indices]
            
        elif self.head_type == "single_categorical":
            all_names = self.all_feature_names["single_categorical"]
            indices = [all_names.index(name) for name in self.selected_features if name in all_names]
            all_single_c_target = raw_tabs[:, numerical_num:(numerical_num + single_c_num)]
            target = all_single_c_target[:, indices]
            
        elif self.head_type == "multi_categorical":
            all_names = list(self.all_feature_names["multi_categorical"].keys())
            selected_names = list(self.selected_features.keys())
            indices = [all_names.index(name) for name in selected_names if name in all_names]
            all_m_c_target = raw_tabs[:, -multi_c_num:]
            target = all_m_c_target[:, indices]
            
        elif self.head_type == "classification":
            target = raw_tabs
            
        return target
