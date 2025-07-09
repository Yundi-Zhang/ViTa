import numpy as np
import torch
from torch import optim, nn
from torchvision.models import resnet18, resnet50
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import wandb
import lightning.pytorch as pl
from networks.tokenizers import *
from utils.losses import BinaryReconCriterion, CategoricalReconCriterion, RegressionCriterion, NumericalReconCriterion

    
class ResNet50Classification(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.img_shape = self.hparams.val_dset[0][0].shape
        self.num_classes = 1
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.img_sample = [] # For image logging
        self.val_gt = []
        self.val_pred = []
        self.val_subject_id = []
        self.save_embeddings = None
        self.save_root_path = None
        
        self.network = self._initial_network()
        # self.regression_criterion = RegressionCriterion(**kwargs)
        self.selected_features = self.hparams.tabular_hparams.selected_features
        self.criterion = BinaryReconCriterion(selected_features=self.selected_features,)
        
        self.test_gt = []
        self.test_pred = []
        self.test_subject_id = []
        self.test_results_path = None
    
    def _initial_network(self):
        S, T = self.img_shape[:2]
        _network = resnet50(pretrained=False, num_classes=self.num_classes)
        _network.conv1 = torch.nn.Conv2d(in_channels=S*T, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        return _network
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.training_hparams.lr)
        return optimizer
    
    def forward(self, x):
        input_shape = x.shape # [B, S, T, H, W]
        x = x.view(input_shape[0], -1, input_shape[-2], input_shape[-1]) # [B, S*T, H, W]
        x = self.network(x)

        return x
    
    def training_step(self, batch, batch_idx, mode="train"):
        imgs, targets, subject_id = batch
        
        pred = self.forward(imgs)
        loss_dict = self.criterion(pred, targets)
        
        # Logging
        for k, v in loss_dict.items():
            if v is None:
                self.log(f"{mode}/{k}", np.nan, on_epoch=True, on_step=False)
            else:
                self.log(f"{mode}/{k}", v, on_epoch=True, on_step=False)

        return loss_dict["loss"]
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, mode="val"):
        imgs, targets, subject_id = batch
        pred = self.forward(imgs) 
        loss_dict = self.criterion(pred, targets.float())
        
        eval(f"self.{mode}_gt").append(targets.detach().cpu())
        eval(f"self.{mode}_pred").append(pred.detach().cpu())
        eval(f"self.{mode}_subject_id").append(subject_id.detach())
        
        for k, v in loss_dict.items():
            self.log(f"{mode}/{k}", v, on_epoch=True, on_step=False)
            
    @torch.no_grad()
    def test_step(self, batch, batch_idx) -> torch.Tensor:
        _ = self.validation_step(batch, batch_idx, mode="test")
    
    def on_validation_epoch_end(self, mode="val") -> None:
        gts = eval(f"torch.concat(self.{mode}_gt, dim=0)")
        subjects_ids = eval(f"torch.concat(self.{mode}_subject_id, dim=0)")
        preds = eval(f"torch.concat(self.{mode}_pred, dim=0)")
        self.log_test_single_categorical_results(gts, preds, mode=mode)
            
        self.val_gt.clear()
        self.val_pred.clear()
        self.val_subject_id.clear()
    
    def on_test_epoch_end(self, mode="val") -> None:
        self.on_validation_epoch_end(mode="test")
        
    def log_test_single_categorical_results(self, gts, logits, mode="test"):
        if mode == "test":
            table = wandb.Table(columns=["Feature name", "accuracy", "f1 score", "auc-roc", "average precision"])
        mean_auc_roc = 0.
        mask = ~torch.isnan(gts)
        for i in range(gts.shape[1]):
            name = self.selected_features[i]
            m = mask[:, i]
            masked_gts = gts[m].to(torch.int64)
            masked_logits = logits[m]
            masked_probs = torch.sigmoid(masked_logits)
            masked_labels = torch.argmax(masked_probs, dim=1)
            
            acc = accuracy_score(masked_gts.numpy(), masked_labels.numpy())
            f1 = f1_score(masked_gts.numpy(), masked_labels.numpy())
            auc_roc = roc_auc_score(masked_gts.numpy(), masked_probs.numpy())
            ap = average_precision_score(masked_gts.numpy(), masked_probs.numpy())
            mean_auc_roc += auc_roc
            if mode == "val":
                self.log(f"val/{name}_acc", acc, on_epoch=True, on_step=False)
                self.log(f"val/{name}_f1", f1, on_epoch=True, on_step=False)
                self.log(f"val/{name}_auc_roc", auc_roc, on_epoch=True, on_step=False)
            if mode == "test":
                table.add_data(name, acc, f1, auc_roc, ap)
        
        if mode == "val": # ckpt monitor
            self.log(f"val_aucroc", mean_auc_roc / gts.shape[1], on_epoch=True, on_step=False)
        if mode == "test":
            wandb.log({"Evaluation_table": table})
        
