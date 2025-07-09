from pathlib import Path
from typing import Optional
from termcolor import colored
import torch

from models.SimCLRTableRecon import SimCLRTabRecon
from models.MultimodalSimCLR import MultimodalSimCLR
from models.classification_models import ViTaBinaryClassification, ResNet50BinaryClassification
from models.reconstruction_models import ReconMAE
from models.regression_models import RegrMAE, ResNet18Module, ResNet50Module
from models.segmentation_models import SegMAE


IMAGING_MODULE_LUT = {"reconstruction": [ReconMAE],
                      "regression": [RegrMAE, ResNet18Module, ResNet50Module],
                      "segmentation": [SegMAE]}
TABULAR_IMAGING_MODULE_LUT = {"pretraining": [MultimodalSimCLR],
                              "reconstruction": [SimCLRTabRecon],
                              "classification": [ViTaBinaryClassification, ResNet50BinaryClassification],} 


def initialize_model(args, params, paths, data_module):
    if args.module == "imaging": return initialize_imaging_model(params, paths, data_module)
    elif args.module == "imaging_tabular": return initialize_imaging_tabular_model(params, paths, data_module)
    else: raise ValueError("We only suppert imaging and imaging tabular models")


# ----------------------------------------------------------------------------------------------------------------------
### Initalizing imaging models ###
def initialize_imaging_model(params, paths, data_module):
    if params.module.task_idx == 0: 
        module_key = "reconstruction"
        module_params = {**params.module.training_hparams.__dict__, **params.module.recon_hparams.__dict__}
    elif params.module.task_idx == 1:
        module_key = "regression"
        module_params = {**params.module.training_hparams.__dict__, **params.module.regr_hparams.__dict__}
    elif params.module.task_idx == 2:
        module_key = "segmentation"
        module_params = {**params.module.training_hparams.__dict__, **params.module.seg_hparams.__dict__}
    else:
        raise NotImplementedError
    module_cls = IMAGING_MODULE_LUT[module_key][params.module.module_idx]
    model = module_cls(val_dset=data_module.val_dset, **module_params)

    # Load pretrained weights into imaging encoder
    if params.general.load_encoder:
        assert params.general.resume_ckpt_path != None, "The path for imaging encoder checkpoint is not provided."
        load_pretrained_weights(model=model, 
                                ckpt_path=Path(paths.log_folder) / params.general.resume_ckpt_path, 
                                freeze_encoder=params.general.freeze_encoder,
                                loading_parts=["encoder_imaging"],
                                exclude_parts=["encoder_imaging.enc_pos_embed"])
    if params.general.load_decoder:
        assert params.general.resume_ckpt_path != None, "The path for imaging decoder checkpoint is not provided."
        load_pretrained_weights(model=model, 
                                ckpt_path=Path(paths.log_folder) / params.general.resume_ckpt_path, 
                                loading_parts=["decoder_imaging"],
                                exclude_parts=["decoder_imaging.enc_pos_embed"])
    return model


# ----------------------------------------------------------------------------------------------------------------------
### Initalizing imaging tabular models ###
def initialize_imaging_tabular_model(params, paths, data_module):
    if params.module.task_idx == 0: module_key = "pretraining"
    elif params.module.task_idx == 1: module_key = "reconstruction"
    elif params.module.task_idx == 2: module_key = "classification"
    else: raise NotImplementedError
    module_cls = TABULAR_IMAGING_MODULE_LUT[module_key][params.module.module_idx]
    model = module_cls(val_dset=data_module.val_dset, 
                       batch_size=data_module._train_dataloader.batch_size, 
                       validation_form=data_module.validation_form,
                       selected_cols=data_module.selected_cols,
                       **params.module.__dict__)
    
    # Load pretrained weights into imaging encoder
    if params.module.imaging_hparams.load_imaging_encoder:
        assert params.module.imaging_hparams.imaging_ckpt_path != None, "The path for imaging encoder checkpoint is not provided."
        load_pretrained_weights(model=model,
                                ckpt_path=Path(paths.log_folder) / params.module.imaging_hparams.imaging_ckpt_path,
                                freeze_encoder=params.module.imaging_hparams.freeze_imaging_encoder,
                                loading_parts=["encoder_imaging"],
                                exclude_parts=["encoder_imaging.enc_pos_embed"])
        
    # Load pretrained weights into imaging decoder
    if params.module.imaging_hparams.load_imaging_decoder:
        assert params.module.imaging_hparams.imaging_ckpt_path != None, "The path for imaging encoder checkpoint is not provided."
        load_pretrained_weights(model=model, 
                                ckpt_path=Path(paths.log_folder) / params.module.imaging_hparams.imaging_ckpt_path, 
                                freeze_encoder=params.module.imaging_hparams.freeze_imaging_encoder,
                                loading_parts=["dec_pos_embed", "decoder", "recon_head"],
                                exclude_parts=["decoder_embed"])
        
    # Load pretrained weights into tabular encoder
    if params.module.tabular_hparams.load_tabular_encoder:
        assert params.module.tabular_hparams.tabular_ckpt_path != None, "The path for tabular encoder checkpoint is not provided."
        load_pretrained_weights(model=model, 
                                ckpt_path=Path(paths.log_folder) / params.module.tabular_hparams.tabular_ckpt_path, 
                                freeze_encoder=params.module.tabular_hparams.freeze_tabular_encoder,
                                loading_parts=["encoder_tabular"])
    
    return model


# ----------------------------------------------------------------------------------------------------------------------
def load_pretrained_weights(model, 
                            ckpt_path: str, 
                            freeze_encoder: bool = False, 
                            loading_parts: Optional[list] = None,
                            exclude_parts: Optional[list] = None,
                            ) -> None:
    """
    Can load imaging encoder with pretrained weights from previous checkpoint/run
    """
    # Load pretraining encoder
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    pretrained_dict = ckpt["state_dict"]
    state_dict_encoder = {}
    if loading_parts is not None:
        for k in list(pretrained_dict.keys()):
            if k.startswith(tuple(loading_parts)):
                if exclude_parts is not None:
                    if not k.startswith(tuple(exclude_parts)):
                        state_dict_encoder[k] = pretrained_dict[k]
                else:
                    state_dict_encoder[k] = pretrained_dict[k]
    else:
        state_dict_encoder = pretrained_dict
    model.load_state_dict(state_dict_encoder, strict=False)

    print(colored(f"Loaded model weights for {loading_parts}", 'yellow', None, ['bold']))
    
    if freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith(tuple(loading_parts)):
                print(name)
                param.requires_grad = False
        print(colored(f"Freeze model weights {loading_parts}", 'yellow', None, ['bold']))
    return
