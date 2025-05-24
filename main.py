from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path

import torch
import wandb

from utils.general import get_data_paths, parser_command_line
from initializers.initialize_dataloader import initialize_dataloader
from initializers.initialize_model import initialize_model
from initializers.initialize_parameters import initialize_parameters

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint

with open('datasets/data_files/tabular_files/feature_names.json', 'r') as f:
    FEATURE_NAMES = json.load(f)

# Check
MONITOR_LUT = {
    "imaging": [("val_PSNR", "model-{epoch:03d}-{val_PSNR:.2f}", "max"), # Reconstruction
                ("val_MAE", "model-{epoch:03d}-{val_MAE:.2f}", "min"), # Regression
                ("val_Dice_FG", "model-{epoch:03d}-{val_Dice_FG:.2f}", "max"), # Segmentation
                ],
    "imaging_tabular": [("val_loss", "model-{epoch:03d}-{val_loss:.2f}", "min"), # Pretraining
                        ("ckpt_metric", "model-{epoch:03d}-{ckpt_metric:.2f}", "min"), # Tabular Reconstruction
                        ],
    }


@rank_zero_only
def get_wandb_logger(wandb_run_name, time_now, args, paths, params):
    wandb_kwargs = dict()
    wandb_kwargs["entity"] = params.general.wandb_entity
    wandb_kwargs["group"] = args.wandb_group_name if args.wandb_group_name is not None else params.general.wandb_group_name
    wandb_kwargs["name"] = f"{wandb_run_name}_{time_now}"
    wandb_kwargs["resume"] = "allow" # Resume if the run_id is provided and identical to a previous run otherwise, start a new run
    if params.general.wandb_run_id is not None:
        wandb_kwargs["id"] = params.general.wandb_run_id

    return WandbLogger(save_dir=paths.log_folder, project=params.general.wandb_project_name, 
                       config=asdict(params), **wandb_kwargs,)


def run():
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("medium")
    
    # Initialize and override args, parameters, and paths
    args = parser_command_line() # Load the arguments from the command line
    paths = get_data_paths() # Get the file path from the .env file
    params = initialize_parameters(args)
    os.environ["WANDB_DISABLED"] = params.general.wandb_disabled
    seed_everything(params.general.seed, workers=True) # Sets seeds for numpy, torch and python.random.

    # Initialize wandb logging
    wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else params.general.wandb_run_name
    time_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    logger = get_wandb_logger(wandb_run_name, time_now, args, paths, params)
    if os.environ.get("RANK", "0") != "0":  # Default rank is 0 for single-GPU training
        os.environ["WANDB_MODE"] = "disabled"

    try:
        if params.module.tabular_hparams.decoder_head_type == "multi_categorical":
            params.module.tabular_hparams.selected_features = FEATURE_NAMES["multi_categorical"]
    except:
        print("This is not multi_categorical")
    
    # Initialize data module
    data_module = initialize_dataloader(args, params, paths)

    # Initialze lighting module
    model = initialize_model(args, params, paths, data_module)
    
    # Check the resuming and loading of the checkpoints
    if params.general.resume_training:  # Resume training
        assert params.general.resume_ckpt_path != None, "The path for checkpoint is not provided."
        resume_ckpt_path = Path(paths.log_folder) / params.general.resume_ckpt_path
        ckpt_dir = resume_ckpt_path.parent
        if wandb_run_name != ckpt_dir.parent.name and wandb_run_name is not None:
            ckpt_dir = resume_ckpt_path.parent.parent.parent / wandb_run_name / time_now
        print(f"ckpt_dir: {ckpt_dir}")
        checkpoint = torch.load(resume_ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        # resume_ckpt_path = None
        ckpt_dir = os.path.join(f"{paths.log_folder}/checkpoints_{args.module}/{wandb_run_name}/{time_now}")
    
    # Monitor foreground dice for segmentation. When reconstruction, monitor PSNR. MAE for regression.
    if args.module == "imaging_tabular" and params.module.task_idx ==0 and params.data.validation_data.validation_form == "classification":
        monitor_metric, ckpt_filename, monitor_mode = ("val_classifier_auc", "model-{epoch:03d}-{val_classifier_auc:.2f}", "max")
    elif args.module == "imaging_tabular" and params.module.task_idx == 1:
        if params.module.tabular_hparams.decoder_head_type in ["classification", "single_categorical", "multi_categorical"]:
            monitor_metric, ckpt_filename, monitor_mode = ("val_aucroc", "model-{epoch:03d}-{val_aucroc:.2f}", "max")
        elif params.module.tabular_hparams.decoder_head_type == "numerical":
            monitor_metric, ckpt_filename, monitor_mode = ("val_mae", "model-{epoch:03d}-{val_mae:.2f}", "min")
    else:
        monitor_metric, ckpt_filename, monitor_mode = MONITOR_LUT[args.module][params.module.task_idx]
            
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename=ckpt_filename, monitor=monitor_metric, 
                                          mode=monitor_mode, save_top_k=1, save_last=True, verbose=True,
                                          every_n_epochs=params.module.training_hparams.val_log_rate, )
    
    # Initialize trainer
    trainer = Trainer(
        default_root_dir=paths.log_folder,
        logger=logger,
        callbacks=[checkpoint_callback,],
        fast_dev_run=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=2,
        benchmark=True,
        profiler="simple",
        strategy="ddp" if params.trainer.devices > 1 else "auto",
        
        **params.trainer.__dict__,
    )

    if args.pipeline == "train":
        model.save_embeddings = False
        trainer.fit(model, datamodule=data_module)
    elif args.pipeline == "val":
        trainer.validate(model, datamodule=data_module)
    elif args.pipeline == "test":
        # if args.module == "imaging_tabular" and params.module.task_idx == 1 and params.module.tabular_hparams.decoder_head_type == "numerical":
        #     model.save_embeddings = True
        #     # model.save_root_path = "embeddings_c1_588_42k.npz"
        #     # value_name = params.data.target_value_name[0].replace("/", "_")
        #     # model.test_results_path = f"results/tabular_features/tab_morphology_age.pkl"
        #     print(f"Saving results in {model.test_results_path}")
        trainer.test(model, datamodule=data_module)
        # if torch.cuda.is_available():
        #     model = model.to("cuda")
        # model.save_seg(data_loader=data_module.test_dataloader())
    elif args.pipeline == "vis":
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.save_embeddings = True
        # model.save_root_path = "embeddings_c1_588_10k_temp.npz"
        model.generate_temp_latents(data_loader=data_module.train_dataloader(),
                            #    token_path=params.data.vis_token_path,
                            #    tsne_map_path=params.data.vis_tsne_map_path,
                            #    save_all_patch_tokens=params.data.save_all_patch_tokens,
                            #    save_tsne=params.data.save_tsne
                               )
        print(f'Latent code saved to {model.save_root_path} and code exits.')
        
    wandb.finish() 


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    run()

