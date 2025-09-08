import json
import random
import time
from typing import List, Tuple, Dict

from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import torch
import lightning.pytorch as pl
import torchmetrics
from termcolor import colored
import wandb

from networks.imaging_encoders import ImagingMaskedEncoder
from utils.clip_loss import CLIPLoss
from lightly.models.modules import SimCLRProjectionHead
from transformers import get_cosine_schedule_with_warmup

from networks.tabular_encoders import *
from utils.logging_related import imgs_to_wandb_video
from utils.utils_visualization import plot_one_label


with open('datasets/data_files/tabular_files/feature_names.json', 'r') as f:
    FEATURE_NAMES_IN = json.load(f)
    

class MultimodalSimCLR(pl.LightningModule):
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
        self.save_embeddings = None
        self.save_root_path = None
        self.training_classifer = None
        # --------------------------------------------------------------------------
        # Imaging
        
        use_both_axes = True if self.hparams.val_dset.get_view() == 2 else False # For positional embedding
        img_shape = self.hparams.val_dset[0][0].shape
        self.S_sax = self.hparams.val_dset.sax_slice_num
        self.encoder_imaging = ImagingMaskedEncoder(img_shape=img_shape, use_both_axes=use_both_axes, 
                                                    S_sax=self.S_sax, **imaging_hp.__dict__)
        self.projector_imaging = SimCLRProjectionHead(imaging_hp.enc_embed_dim, 
                                                      imaging_hp.enc_embed_dim, 
                                                      imaging_hp.projection_dim)
        # --------------------------------------------------------------------------
        # Tabular
        self.tabular_encoder_cls = globals()[tabular_hp.tabular_encoder_type]
        self.encoder_tabular = self.tabular_encoder_cls(tabular_hp, 
                                                        patch_num=self.encoder_imaging.patch_embed.num_patches,
                                                        all_feature_names=FEATURE_NAMES_IN)
        self.projector_tabular = SimCLRProjectionHead(tabular_hp.embedding_dim, 
                                                      tabular_hp.embedding_dim, 
                                                      tabular_hp.projection_dim)
        # --------------------------------------------------------------------------
        # Criterion
        self.criterion = CLIPLoss(temperature=self.hparams.training_hparams.temperature, 
                                  lambda_0=self.hparams.training_hparams.lambda_0)
        # --------------------------------------------------------------------------
        # Classifier
        ### Accuracy calculated against all others in batch of same view except for self and all of the other view
        cl_nclasses = self.hparams.batch_size
        self.top1_acc = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=cl_nclasses)
        if cl_nclasses >= 4:
            self.top4_acc = torchmetrics.Accuracy(task='multiclass', top_k=4, num_classes=cl_nclasses)
        ### Accuracy calculated for classification evaluation
        metric_nclasses = self.hparams.training_hparams.num_classes
        task = 'binary' if metric_nclasses == 2 else 'multiclass'
        self.classifier_acc = torchmetrics.Accuracy(task=task, num_classes=metric_nclasses)
        self.classifier_auc = torchmetrics.AUROC(task=task, num_classes=metric_nclasses)
        # --------------------------------------------------------------------------
        print(colored('Tabular model, multimodal:', 'yellow', None, ['bold']), f'{self.encoder_tabular}\n{self.projector_tabular}')
        print(colored('Imaging model, multimodal:', 'yellow', None, ['bold']), f'{self.encoder_imaging}\n{self.projector_imaging}')
        
    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.time()
    
    def configure_optimizers(self) -> Tuple[Dict, Dict]:
        """
        Define and return optimizer and scheduler for contrastive model. 
        """
        optimizer = torch.optim.Adam(
            [{'params': self.encoder_imaging.parameters()}, 
             {'params': self.projector_imaging.parameters()},
             {'params': self.encoder_tabular.parameters()},
             {'params': self.projector_tabular.parameters()}], 
            
            lr=self.hparams.training_hparams.lr, 
            weight_decay=self.hparams.training_hparams.weight_decay,)
        scheduler = self.initialize_scheduler(optimizer)
        return ({"optimizer": optimizer, "lr_scheduler": scheduler})

    def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.hparams.training_hparams.scheduler == 'cosine':
            T_max = int(self.hparams.training_hparams.dataset_length*self.hparams.training_hparams.cosine_anneal_mult)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0, last_epoch=-1)
        elif self.hparams.training_hparams.scheduler == 'anneal':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.training_hparams.warmup_epochs,
                num_training_steps=self.hparams.training_hparams.anneal_max_epochs
            )
        else:
            raise ValueError('Valid schedulers are "cosine" and "anneal"')
        return scheduler

    def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates projection and encoding of imaging data.
        """
        y, _, _ = self.encoder_imaging(x)
        if self.hparams.training_hparams.avg_token:
            x_img = y.mean(dim=1)
        else:
            x_img = y[:, 0, :] # Take the class token
        z_img = self.projector_imaging(x_img)
        return z_img, x_img, y
    
    def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates projection and encoding of tabular data.
        """
        y = self.encoder_tabular(x)
        if self.hparams.training_hparams.avg_token:
            x_tab = y.mean(dim=1)
        else:
            x_tab = y[:, 0, :] # Take the class token
        z_tab = self.projector_tabular(x_tab)
        return z_tab, x_tab
    
    def forward(self, imgs, tabs):
        image_features, _, _ = self.forward_imaging(imgs)
        table_features, _ = self.forward_tabular(tabs)
        return image_features, table_features
    
    def training_step(self, batch, batch_idx, mode="train") -> torch.Tensor:
        """
        Trains contrastive model
        """
        imgs, tabs, y, subject_ids = batch

        projected_img_embeddings, img_embeddings, all_img_embeddings = self.forward_imaging(imgs) 
        projected_tab_embeddings, tab_embeddings = self.forward_tabular(tabs)
        loss, logits, labels = self.criterion(projected_img_embeddings, projected_tab_embeddings)
        if mode in ["train", "val"]:
            eval(f"self.{mode}_step_outputs").append({'img_embeddings': img_embeddings.detach().cpu(), 
                                                    'tab_embeddings': tab_embeddings.detach().cpu(), 
                                                    'labels': y.detach().cpu()})
        else:
            eval(f"self.test_step_outputs").append({'img_embeddings': img_embeddings.detach().cpu(), 
                                                    'all_img_embeddings': all_img_embeddings.detach().cpu(), 
                                                    'tab_embeddings': tab_embeddings.detach().cpu(), 
                                                    'subject_ids': subject_ids.detach().cpu(), 
                                                    'labels': y.detach().cpu()})
        
        self.log(f"{mode}/loss", loss.item(), on_epoch=True, on_step=False)
        if imgs.shape[0] == self.hparams.batch_size:
            top1_acc = self.top1_acc(logits, labels)
            self.log(f"{mode}/top1", top1_acc.item(), on_epoch=True, on_step=False, prog_bar=True)
            if self.hparams.batch_size >= 4:
                top4_acc = self.top4_acc(logits, labels)
                self.log(f"{mode}/top4", top4_acc, on_epoch=True, on_step=False)
        
        if mode == "val":
            self.log(f"val_loss", loss.item(), on_epoch=True, on_step=False) # For monitor
            if self.hparams.training_hparams.log_images:
                self.img_sample.append(imgs[0].detach().cpu())
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        _ = self.training_step(batch, batch_idx, mode="val")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        _ = self.training_step(batch, batch_idx, mode="test")
    
    def on_train_epoch_end(self, mode="train") -> None:
        """
        Train and log classifier
        """
        # Only log based on logging rate
        if (self.current_epoch + 1) % eval(f"self.hparams.training_hparams.{mode}_log_rate") == 0:
            if eval(f"len(self.{mode}_step_outputs)") == 0: 
                return
            if mode == "val" and len(self.train_step_outputs) == 0: 
                return

            # Log metrics for validation
            if self.hparams.validation_form == "classification": 
                self.log_classification_metrics(mode=mode)
            elif self.hparams.validation_form == "visualization": 
                self.log_visualization_metrics(mode=mode)
            else: 
                raise NotImplementedError
    
    def on_validation_epoch_end(self) -> None:
        self.on_train_epoch_end(mode="val")
        
    def on_test_epoch_end(self) -> None:
        self.hparams.training_hparams.test_log_rate = 1
        self.on_train_epoch_end(mode="test")
        
    def log_classification_metrics(self, mode="train"):
        train_img_embeddings, train_tab_embeddings, train_labels = self.stack_outputs(self.train_step_outputs)
        if len(torch.unique(train_labels)) == 1:
            print("The labels for loaded data are the same, can't calculate classifier. Logging is being skipped")
            return
        
        self.training_classifer = LogisticRegression(class_weight='balanced', max_iter=1000).fit(train_img_embeddings, train_labels)
        if mode == "train":
            eval_embeddings = train_img_embeddings
            eval_labels = train_labels
        elif mode in ["val", "test"]:
            img_embeddings, tab_embeddings, labels = eval(f"self.stack_outputs(self.{mode}_step_outputs)")
            eval_embeddings = img_embeddings
            eval_labels = labels
        
        preds, probs = self.predict_live_training_classifer(eval_embeddings)
        acc = self.classifier_acc(preds, eval_labels).item()
        auc = self.classifier_auc(probs, eval_labels).item()
        self.log(f'{mode}/classifier_acc', acc, on_epoch=True, on_step=False)
        self.log(f'{mode}/classifier_auc', auc, on_epoch=True, on_step=False)
        eval(f"self.{mode}_step_outputs.clear()")
        
        if mode == "val":
            self.log(f'val_classifier_auc', auc, on_epoch=True, on_step=False) # For monitoring
            if self.hparams.training_hparams.log_images:
                idx = random.randint(0, len(self.img_sample) - 1)
                img_sample = self.img_sample[random.randint(0, len(self.img_sample) - 1)]
                image_sample = imgs_to_wandb_video(img_sample)
                self.logger.experiment.log({"Video Example": wandb.Video(image_sample, caption=f"Validation Epoch {self.current_epoch}")})
                self.img_sample.clear()
        if mode == "train":
            epoch_runtime = (time.time() - self.train_epoch_start_time) / 60
            self.log('train/epoch_runtime', epoch_runtime, on_epoch=True, on_step=False)
        return

    def log_visualization_metrics(self, mode="train"):
        if mode == "val":
            img_embeddings, tab_embeddings, labels = self.stack_outputs(self.val_step_outputs)
            scaled_cls = StandardScaler().fit_transform(img_embeddings)
            del embeddings
            tsne_map_cls = TSNE(n_components=3, perplexity=5, learning_rate="auto").fit_transform(scaled_cls)
            for i, feature_name in enumerate(self.hparams.selected_cols):
                name = feature_name.split()[0]
                tsne_plt = plot_one_label(map=tsne_map_cls, map_type="tsne", labels=labels[:, i], label_name=name, dim=2)
                self.logger.experiment.log({f"TSNE_vis_{name}": wandb.Image(tsne_plt, caption=f"TSNE Visualization at training epoch {self.current_epoch}")})
                del tsne_plt
            del scaled_cls, tsne_map_cls
        eval(f"self.{mode}_step_outputs.clear()")
        return

    def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack outputs from multiple steps
        """
        labels, img_embeddings, tab_embeddings = [], [], []
        for i in outputs:
            img_embeddings.append(i["img_embeddings"])
            tab_embeddings.append(i["tab_embeddings"])
            labels.append(i["labels"])
        img_embeddings = torch.concat(img_embeddings, dim=0)
        tab_embeddings = torch.concat(tab_embeddings, dim=0)
        labels = torch.concat(labels, dim=0)
        return img_embeddings, tab_embeddings, labels

    def predict_live_training_classifer(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using live training_classifer
        """
        preds = self.training_classifer.predict(embeddings)
        probs = self.training_classifer.predict_proba(embeddings)
        preds = torch.tensor(preds)
        probs = torch.tensor(probs)
        # Only need probs for positive class in binary case
        if self.hparams.training_hparams.num_classes == 2:
            probs = probs[:,1]
        return preds[:, None], probs[:, None]
