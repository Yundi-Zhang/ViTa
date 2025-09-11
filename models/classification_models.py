import time
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from termcolor import colored
from transformers import get_cosine_schedule_with_warmup
import torchmetrics
import wandb
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from torchvision.models import resnet50

from networks.imaging_encoders import ImagingMaskedEncoder
from networks.tabular_decoders import TabularDecoder


class ViTaBinaryClassification(pl.LightningModule):
    """
    Lightning module for binary classification using ViTa.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters() # Save all parameters to self.hparams
        
        self.val_probs = []
        self.val_labels = []
        self.test_preds = []
        self.test_probs = []
        self.test_labels = []
        self.best_val_threshold = 0.5

        # Metrics: compute using logits or probabilities
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_ap = torchmetrics.classification.AveragePrecision(task="binary")

        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")
        self.test_ap = torchmetrics.classification.AveragePrecision(task="binary")

        # Imaging encoder
        use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False # For positional embedding
        img_shape = self.hparams.val_dset[0][0].shape
        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, 
                                                    use_both_axes=use_both_axes, 
                                                    **self.hparams.imaging_hparams.__dict__)

        # Tabular decoder
        self.decoder_type = self.hparams.tabular_hparams.decoder_type
        if self.decoder_type in ["linear", "linear_cls"]:
            if self.decoder_type == "linear":
                tabular_input_dim = self.encoder_imaging.patch_embed.out_channels * self.encoder_imaging.patch_embed.num_patches
            elif self.decoder_type == "linear_cls":
                tabular_input_dim = self.encoder_imaging.patch_embed.out_channels
            self.decoder_tabular = TabularDecoder(input_dim=tabular_input_dim, 
                                                  out_dim=self.hparams.tabular_hparams.decoder_dim,
                                                  dim=self.hparams.tabular_hparams.decoder_dim, 
                                                  depth=self.hparams.tabular_hparams.decoder_depth)
        elif self.decoder_type in ["pool", "pool_cls"]:
            self.decoder_tabular = torch.nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_features=self.hparams.tabular_hparams.decoder_dim, out_features=1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.tabular_hparams.pos_weight).to(self.device))

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
        if self.decoder_type.split('_')[-1] == "cls":
            x = all_tokens[:, 0] # class token
        else:
            x = all_tokens[:, 1:] # all tokens for patches
        x = self.decoder_tabular(x)
        x = self.head(x)
        return x.squeeze(1)
    
    def training_step(self, batch, batch_idx, mode="train") -> torch.Tensor:

        imgs, y, subject_id = batch
        y = y.squeeze(1)
        logits = self.forward(imgs) 
        loss = self.criterion(logits, y)
        self.log(f"{mode}/loss", loss, on_epoch=True, on_step=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, mode="val"):
        imgs, y, subject_id = batch
        y = y.squeeze(1)
        logits = self.forward(imgs) 
        probs = torch.sigmoid(logits)
        preds = (probs > self.best_val_threshold).int()

        loss = self.criterion(logits, y)

        # Update and log metrics
        self.val_probs.extend(probs.cpu().numpy())
        self.val_labels.extend(y.cpu().numpy())

        self.val_precision.update(preds, y.int())
        self.val_recall.update(preds, y.int())
        self.val_f1.update(preds, y.int())
        self.val_auc.update(probs, y.int())
        self.val_ap.update(probs, y.int())

        self.log(f"{mode}/loss", loss, on_epoch=True, on_step=False)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        imgs, y, subject_id = batch
        y = y.squeeze(1)
        logits = self.forward(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > self.best_val_threshold).int()

        self.test_precision.update(preds, y.int())
        self.test_recall.update(preds, y.int())
        self.test_f1.update(preds, y.int())
        self.test_auc.update(probs, y.int())
        self.test_ap.update(probs, y.int())

        # Save for table
        self.test_preds.extend(preds.cpu().numpy())
        self.test_probs.extend(probs.cpu().numpy())
        self.test_labels.extend(y.cpu().numpy())
    
    def on_validation_epoch_end(self):
        self.log("val/precision", self.val_precision.compute(), prog_bar=True)
        self.log("val/recall", self.val_recall.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.log("val/ap", self.val_ap.compute(), prog_bar=True)
        self.log(f"val_f1", self.val_f1.compute(), on_epoch=True, on_step=False) # ckpt monitor

        # --- PR Curve & Best Threshold ---
        if len(self.val_probs) > 0:
            probs = np.array(self.val_probs)
            labels = np.array(self.val_labels)
            precision, recall, thresholds = precision_recall_curve(labels, probs)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            best_idx = np.argmax(f1_scores)
            best_thresh = thresholds[best_idx]
            pr_auc = auc(recall, precision)

            self.log("val/best_threshold", best_thresh)
            self.log("val/pr_auc", pr_auc)

            # Plot PR curve
            plt.figure()
            plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True)

            # Log to wandb
            wandb.log({"val/pr_curve": wandb.Image(plt)})
            plt.close()

            self.best_val_threshold = best_thresh

        # Reset for next epoch
        self.val_probs.clear()
        self.val_labels.clear()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auc.reset()
        self.val_ap.reset()


    def on_test_epoch_end(self):
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        f1 = self.test_f1.compute()
        auc = self.test_auc.compute()
        ap = self.test_ap.compute()
        acc = (torch.tensor(self.test_preds) == torch.tensor(self.test_labels)).float().mean()

        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auc.reset()
        self.test_ap.reset()

        # Table logging
        test_table = wandb.Table(columns=["Feature name", "auc-roc", "f1 score", "recall", "precision", "Average Precision"])

        test_table.add_data(self.hparams.tabular_hparams.selected_features[0], auc, f1, recall, precision, ap)
        wandb.log({"Evaluation_table": test_table})


class ResNet50BinaryClassification(ViTaBinaryClassification):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.val_probs = []
        self.val_labels = []
        self.test_preds = []
        self.test_probs = []
        self.test_labels = []
        self.best_val_threshold = 0.5

        # Metrics: compute using logits or probabilities
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_ap = torchmetrics.classification.AveragePrecision(task="binary")

        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")
        self.test_ap = torchmetrics.classification.AveragePrecision(task="binary")

        # Network
        self.img_shape = self.hparams.val_dset[0][0].shape
        self.num_classes = 1
        
        self.network = self._initial_network()
        self.selected_features = self.hparams.tabular_hparams.selected_features
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.hparams.tabular_hparams.pos_weight).to(self.device))

    def _initial_network(self):
        S, T = self.img_shape[:2]
        _network = resnet50(pretrained=False, num_classes=self.num_classes)
        _network.conv1 = torch.nn.Conv2d(in_channels=S*T, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        return _network
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.training_hparams.lr)
        return optimizer
    
    def forward(self, x):
        B, S, T, H, W = x.shape # (B, S, T, H, W)
        x = x.view(B, -1, H, W) # (B, S*T, H, W)
        x = self.network(x) # (B, 1)

        return x.squeeze(1)
