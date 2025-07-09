from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Optional
from monai.losses import DiceLoss
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from monai.metrics import compute_hausdorff_distance


from utils.general import get_data_paths, to_1hot


def psnr(img, ref, axes = (0, 1), max_intensity=None, reduction="mean"):
    """ Compute the peak signal to noise ratio (psnr)
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the psnr is computed
    :param max_intensity: maximum intensity in the image. If it is None, the maximum value of :ref: is taken.
    :return: (mean) psnr
    """
    mse = np.mean(np.abs(np.abs(img) - np.abs(ref))**2, axis=axes)
    max_intensity = np.max(np.abs(ref)) if max_intensity == None else max_intensity
    mse = 10 * np.log10(max_intensity ** 2 / mse)
    if reduction == "mean":
        psnr_values = np.mean(mse)
    elif reduction == "none":
        psnr_values = mse
    return psnr_values


def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None, 
                   replace_with_gt: bool = False,
                   reduction="mean"):
    """To calculate the psnr of the given prediction. If replace with gt is True, the prediction will be replaced by the ground truth value, but only where the mask is 0"""
    if replace_with_gt:
        assert mask is not None, "The mask is not provided to psnr calculation with replacement"
        if mask.shape != pred.shape:
            mask = mask.view(*mask.shape, 1)
            mask = mask.repeat((1, 1, pred.shape[-1]))
        pred = torch.where(mask == 0, gt, pred)
    psnr_value = psnr(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), axes=(-2, -1), reduction=reduction)
    return psnr_value
    

class PSNR(torch.nn.Module):
    def __init__(self, max_value=1.0, magnitude_psnr=True):
        super(PSNR, self).__init__()
        self.max_value = max_value
        self.magnitude_psnr = magnitude_psnr

    def forward(self, u, g, mask=None):
        """

        :param u: noised image
        :param g: ground-truth image
        :param mask: mask for the image
        :return:
        """
        if self.magnitude_psnr:
            u, g = torch.abs(u), torch.abs(g)
        batch_size = u.shape[0]
        diff = (u.reshape(batch_size, -1) - g.reshape(batch_size, -1))
        if mask is not None:
            diff = diff[mask.reshape(batch_size, -1) == 1]
        square = torch.conj(diff) * diff
        max_value = g.abs().max() if self.max_value == "on_fly" else self.max_value
        if square.is_complex():
            square = square.real
        v = torch.mean(20 * torch.log10(max_value / torch.sqrt(torch.mean(square, -1))))
        return v
    

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CosineSimilarity(dim=1)
        
    # def forward(self, p1, p2, z1, z2):
    #     """y is detached from the computation of the gradient.
    #     This is inspired by paper: Chen and He, et al. 2020. Exploring simple siamese representation learning."""
    #     return -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
    
    def forward(self, p2, z1):
        """y is detached from the computation of the gradient.
        This is inspired by paper: Chen and He, et al. 2020. Exploring simple siamese representation learning."""
        return -self.criterion(p2, z1).mean()
    
    
class ReconstructionCriterion(torch.nn.Module):
    def __init__(self, loss_types, loss_weights: Optional[list[float]] = None, **kwargs):
        super().__init__()
        self.max_value = 1.0
        self.loss_types = loss_types
        self.loss_weights = loss_weights if loss_weights is not None else [1.0] * len(self.loss_types)
        self.loss_fcts = []
        loss_fct_dict = {"mse": torch.nn.MSELoss(reduction="none"), "cl": ContrastiveLoss()}
        if not isinstance(self.loss_types, list):
            self.loss_fcts = [loss_fct_dict[self.loss_types]]
        else:
            self.loss_fcts += [loss_fct_dict[loss_name] for loss_name in self.loss_types]

        self.use_mask_in_loss = kwargs.get("mask_loss", False)
        
    def forward(self, x, y, mask: Optional[torch.Tensor] = None, **kwargs):
        """Compute reconstruction loss
        x: [B, L, D] torch.Tensor, reconstructed image
        y: [B, L, D] torch.Tensor, reference image
        mask: [B, L] torch.Tensor, mask for encoder, where 0 is keep, 1 is remove
        
        loss_dict: dict, loss values
        psnr_value: float, psnr value
        """
        assert x.shape[-1] == y.shape[-1]
        if self.use_mask_in_loss:
            masked_x = x[mask == 1]
            masked_y = y[mask == 1]
        else:
            masked_x = x
            masked_y = y
        # --------------------------------------------------------------------------
        # Calculate losses
        total_loss = 0.0
        loss_dict = {}
        for i, loss_name in enumerate(self.loss_types):
            self.loss_fct = self.loss_fcts[i]
            if loss_name == "cl":
                p2, z1 = kwargs["p2"], kwargs["z1"]
                loss = self.loss_fct(p2, z1)
                # p1, p2, z1, z2 = kwargs["p1"], kwargs["p2"], kwargs["z1"], kwargs["z2"]
                # loss = self.loss_fct(p1, p2, z1, z2)
            else:    
                loss = self.loss_fct(masked_x, masked_y)
                loss = loss.mean()
            loss_dict[loss_name] = loss
            total_loss += self.loss_weights[i] * loss
        loss_dict["loss"] = total_loss
        return loss_dict
    
    
class RegressionCriterion(torch.nn.Module):
    def __init__(self, loss_types, **kwargs):
        super().__init__()
        assert len(loss_types) == 1
        for type in loss_types:
            if type == "mse":
                self.loss_fct = torch.nn.MSELoss(reduction="mean")
            elif type == "huber":
                self.loss_fct = torch.nn.HuberLoss(reduction="mean")
            else:
                raise NotImplementedError("Loss function {} is not implemented for regression".format(type))
    
    def forward(self, pred, target):
        loss = 0.0
        mask = ~torch.isnan(target)
        loss = self.loss_fct(pred[mask], target[mask])
        mae = torch.abs(pred[mask] - target[mask]).detach().mean().item()
        return loss, mae
    

class SegmentationCriterion(torch.nn.Module):
    def __init__(self, num_classes, data_view, loss_types, **kwargs):
        super().__init__()
        assert loss_types[0] == "dice" 

        self.num_classes = num_classes
        self.data_view = data_view
        self.loss_fct = DiceLoss(reduction="none")
    
    def forward(self, pred, target, mode="train"):
        """Input size: (B, C, S, T, H, W)"""
        pred = pred.moveaxis(1, 2)
        target = target.moveaxis(1, 2)
        B, S = pred.shape[:2]
        slice_mask = torch.ones((B, S), dtype=torch.bool)
        if self.data_view != 0: # If it is either long-axis or both axes
            slice_mask[:, 1] = 0
        pred_ = pred[slice_mask] # (S, C, T, H, W)
        target_ = target[slice_mask] # (S, C, T, H, W)

        loss = self.loss_fct(pred_, target_)
        loss = loss.squeeze().mean(dim=0)
        dice = 1 - loss.detach()

        return loss.mean(), dice


class NumericalReconCriterion(torch.nn.Module):
    def __init__(self, loss_weights, selected_features, use_scalor=False, **kwargs):
        super().__init__()
        self.loss_weights = loss_weights
        self.selected_features = selected_features
        self.use_scalor = use_scalor
        if len(loss_weights) != len(selected_features):
            self.loss_weights = [1.] * len(selected_features)

        if "mse" in kwargs.get("loss_types"):
            self.loss_fct = torch.nn.MSELoss(reduction="mean")
        elif "huber" in kwargs.get("loss_types"):
            self.loss_fct = torch.nn.HuberLoss(reduction="mean")
        
        # Get scaler to transform back to original size
        if self.use_scalor:
            self.mean_values, self.std_values = self.get_column_scaler(selected_features)
            
    def get_column_scaler(self, selected_features):
        paths = get_data_paths()
        df = pd.read_csv(Path(paths.dataloader_tabular_file_folder) / "raw_tab_paul.csv")
        mean_values = df[selected_features].mean()
        std_values = df[selected_features].std()
        return mean_values, std_values
        
    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        losses = {}
        all_mae, all_loss = [], []
        for i in range(target.shape[1]):
            m = mask[:, i]
            if (~m).all():
                loss, mae, r2 = None, np.nan, np.nan
            else:
                masked_pred = pred[:, i][m]
                masked_target = target[:, i][m]
                loss = self.loss_fct(masked_pred, masked_target)
                mae = torch.abs(masked_pred - masked_target).detach().mean().item()
                if self.use_scalor: # For scaled target values
                    mae = mae * self.std_values[i]
                r2 = r2_score(masked_target.detach().cpu(), masked_pred.detach().cpu())
                all_loss.append(self.loss_weights[i] * loss)
                all_mae.append(mae)
            losses[f"mae_{self.selected_features[i]}"] = mae
            losses[f"r2_{self.selected_features[i]}"] = r2
        # loss #####
        if len(all_loss) == 0: 
            losses["loss"] = None
            losses["mean_mae"] = None
        else: 
            losses["loss"] = sum(all_loss)
            losses["mean_mae"] = np.mean(all_mae)
        return losses
    

class CategoricalReconCriterion(torch.nn.Module):
    def __init__(self, loss_weights, selected_features, multi_classes=False, **kwargs):
        super().__init__()
        self.loss_weights = loss_weights
        self.multi_classes = multi_classes
        self.selected_features = selected_features
        if len(loss_weights) != len(selected_features):
            self.loss_weights = [1.] * len(selected_features)

        self.loss_fctn = torch.nn.CrossEntropyLoss()
        
    def compute_metrics(self, y_true, y_pred_logits):
        # y_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        # auc = roc_auc_score(y_true, y_probs[:, 1])
        return acc
    
    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        losses = {}
        weighted_loss = 0
        for i in range(target.shape[1]):
            m = mask[:, i]
            if isinstance(self.selected_features, dict):
                masked_pred = pred[i][m]
            else:
                masked_pred = pred[:, i][m]
            masked_target = target[:, i][m].long()
            loss = self.loss_fctn(masked_pred, masked_target)
            weighted_loss += self.loss_weights[i] * loss
            acc = self.compute_metrics(masked_target, masked_pred)
            for metic_name, metric in zip(["acc"], [acc]):
            # for metic_name, metric in zip(["acc", "auc"], [acc, auc]):
                if isinstance(self.selected_features, dict):
                    log_name = f"{metic_name}_{list(self.selected_features.keys())[i]}"
                else:
                    log_name = f"{metic_name}_{self.selected_features[i]}"
                losses[log_name] = metric
        losses["loss"] = weighted_loss
        return losses
    

class BinaryReconCriterion(torch.nn.Module):
    def __init__(self, selected_features, **kwargs):
        super().__init__()

        self.selected_features = selected_features
        self.loss_fctn = nn.BCEWithLogitsLoss()
        
    def compute_metrics(self, y_true, y_pred_logits):
        # y_probs = torch.sigmoid(y_pred_logits).cpu().numpy()
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        # auc = roc_auc_score(y_true, y_probs[:, 1])
        # Compute for minority class (class 1)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        return acc, precision, recall, f1
    
    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        losses = {}
        m = mask[:, 0]
        masked_pred = pred[m]
        masked_target = target[m]
        loss = self.loss_fctn(masked_pred, masked_target)
        acc, precision, recall, f1 = self.compute_metrics(masked_target, masked_pred)
        losses[f"acc_{self.selected_features[0]}"] = acc
        losses[f"precision_{self.selected_features[0]}"] = precision
        losses[f"recall_{self.selected_features[0]}"] = recall
        losses[f"f1_{self.selected_features[0]}"] = f1
        losses["loss"] = loss
        return losses
