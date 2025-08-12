from dataclasses import asdict, fields
from pathlib import Path
import yaml
from typing import Any, Dict
from datetime import datetime
import os
import wandb
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from configs import imaging_params
from configs import imaging_tabular_params


MONITOR_LUT = {
    "imaging": [("val_PSNR", "model-{epoch:03d}-{val_PSNR:.2f}", "max"), # Reconstruction
                ("val_MAE", "model-{epoch:03d}-{val_MAE:.2f}", "min"), # Regression
                ("val_Dice_FG", "model-{epoch:03d}-{val_Dice_FG:.2f}", "max"), # Segmentation
                ],
    "tabular": [("val_classifier_auc", "model-{epoch:03d}-{val_classifier_auc:.2f}", "max")],
    "imaging_tabular": [("val_loss", "model-{epoch:03d}-{val_loss:.2f}", "min"), # Pretraining
                        ("ckpt_metric", "model-{epoch:03d}-{ckpt_metric:.2f}", "min"), # Tabular Reconstruction
                        ],
    }


@rank_zero_only
def initialize_wandb_logger(args, paths, params):
    os.environ["WANDB_DISABLED"] = params.general.wandb_disabled
    if os.environ.get("RANK", "0") != "0":  # Default rank is 0 for single-GPU training
        os.environ["WANDB_MODE"] = "disabled"

    wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else params.general.wandb_run_name
    time_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    wandb_kwargs = dict()
    wandb_kwargs["entity"] = params.general.wandb_entity
    wandb_kwargs["group"] = args.wandb_group_name if args.wandb_group_name is not None else params.general.wandb_group_name
    wandb_kwargs["name"] = f"{wandb_run_name}_{time_now}"
    wandb_kwargs["resume"] = "allow" # Resume if the run_id is provided and identical to a previous run otherwise, start a new run
    if params.general.wandb_run_id is not None:
        wandb_kwargs["id"] = params.general.wandb_run_id
    logger =  WandbLogger(save_dir=paths.log_folder, project=params.general.wandb_project_name, 
                          config=asdict(params), **wandb_kwargs,)
    
    return logger, wandb_run_name, time_now


def setup_ckpt_path(args, paths, params, wandb_run_name, time_now):
    resume_ckpt_path = None
    if params.general.resume_training:  # Resume training
        assert params.general.resume_ckpt_path != None, "The path for checkpoint is not provided."
        resume_ckpt_path = Path(paths.log_folder) / params.general.resume_ckpt_path
        ckpt_dir = resume_ckpt_path.parent
        if wandb_run_name != ckpt_dir.parent.name and wandb_run_name is not None:
            ckpt_dir = resume_ckpt_path.parent.parent.parent / wandb_run_name / time_now
        print(f"ckpt_dir: {ckpt_dir}")
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
    else:
        ckpt_dir = os.path.join(f"{paths.log_folder}/checkpoints_{args.module}/{wandb_run_name}/{time_now}")
    return ckpt_dir, resume_ckpt_path


def initialize_ckpt_args(args, params):
    #----------- Imaging tabular modules -----------#
    if args.module == "imaging_tabular":
        if params.module.task_idx == 0 and params.data.validation_data.validation_form == "classification":
            #----------- Pretraining -----------#
            monitor_metric, ckpt_filename, monitor_mode = ("val_classifier_auc", "model-{epoch:03d}-{val_classifier_auc:.2f}", "max")
    
        elif params.module.task_idx == 1: 
            #----------- Reconstruction -----------#
            if params.module.tabular_hparams.decoder_head_type in ["single_categorical", "multi_categorical"]:
                monitor_metric, ckpt_filename, monitor_mode = ("val_aucroc", "model-{epoch:03d}-{val_aucroc:.2f}", "max")
            elif params.module.tabular_hparams.decoder_head_type == "numerical":
                monitor_metric, ckpt_filename, monitor_mode = ("val_mae", "model-{epoch:03d}-{val_mae:.2f}", "min")
    
        elif params.module.task_idx == 2:
            #----------- Classification -----------#
            monitor_metric, ckpt_filename, monitor_mode = ("val_f1", "model-{epoch:03d}-{val_f1:.2f}", "max")
    
    #----------- Imaging modules -----------#
    else:
        monitor_metric, ckpt_filename, monitor_mode = MONITOR_LUT[args.module][params.module.task_idx]
        
    return monitor_metric, ckpt_filename, monitor_mode


def initialize_parameters(args):
    config_path = args.config
    # Override the default parameters by the given configuration file
    if args.module == "imaging":
        params = load_imaging_model_config_from_yaml(config_path)
    elif args.module in ["tabular", "imaging_tabular"]:
        params = load_imaging_tabular_model_config_from_yaml(config_path)
    else:
        raise ValueError("We only support imaging or imaging_tabular module")
    return params


def load_imaging_model_config_from_yaml(file_path):
    config_data = dict()
    if file_path is not None:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

    # Get the default values from the data class
    params = imaging_params.Params(general=imaging_params.GeneralParams(), 
                                   data=imaging_params.DataParams(), 
                                   trainer=imaging_params.TrainerParams(), 
                                   module=imaging_params.ModuleParams(
                                       recon_hparams=imaging_params.ReconMAEParams(),
                                       seg_hparams=imaging_params.SegMAEParams(),
                                       regr_hparams=imaging_params.RegrMAEParams(),
                                       training_hparams=imaging_params.TrainingParams()))
    update_params = update_dataclass_from_dict(params, config_data)

    return update_params


def load_imaging_tabular_model_config_from_yaml(file_path):
    config_data = dict()
    if file_path is not None:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

    # Get the default values from the data class
    params = imaging_tabular_params.Params(general=imaging_tabular_params.GeneralParams(), 
                                           data=imaging_tabular_params.DataParams(
                                               general_data=imaging_tabular_params.GeneralDataParams(),
                                               imaging_data=imaging_tabular_params.ImagingDataParams(),
                                               tabular_data=imaging_tabular_params.TabularDataParams(), 
                                               validation_data=imaging_tabular_params.ValidationDataParams()), 
                                           trainer=imaging_tabular_params.TrainerParams(), 
                                           module=imaging_tabular_params.ModuleParams(
                                               imaging_hparams=imaging_tabular_params.ImagingParams(),
                                               tabular_hparams=imaging_tabular_params.TabularParams(),
                                               training_hparams=imaging_tabular_params.TrainingParams()))
    update_params = update_dataclass_from_dict(params, config_data)

    return update_params


def update_dataclass_from_dict(params, config_data: Dict[str, Any]):
    updated_fields = {}
    instance_dict = asdict(params)
    for key in config_data:
        if is_field_name(params, key):
            value = config_data[key]
            if isinstance(value, dict) and hasattr(getattr(params, key), '__dataclass_fields__'):
                # Recursively update nested dataclass
                updated_value = update_dataclass_from_dict(getattr(params, key), value)
                updated_fields[key] = updated_value
            else:
                updated_fields[key] = value
            instance_dict.update(updated_fields)
        else:
            raise NameError(f"{key} is not defined in the dataclass")
    return params.__class__(**instance_dict)


def is_field_name(dataclass_type, field_name):
    return field_name in [f.name for f in fields(dataclass_type)]
    