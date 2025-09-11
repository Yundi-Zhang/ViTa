"""Parameters for the module in a nested dataclass manner."""

from dataclasses import asdict, dataclass, fields
import yaml
from typing import Any, Dict, Optional, Tuple


@dataclass
class GeneralParams:
    # wandb logging configs
    wandb_project_name: str = "MAE"
    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_entity: str = "none"
    wandb_disabled: str = "true"
    
    seed: int = 1
    gpu: int = 0
    resume_training: bool = False
    freeze_encoder: bool = False
    load_encoder: bool = False
    load_decoder: bool = False
    resume_ckpt_path: str = None

    
@dataclass
class DataParams:
    cmr_path_pickle_name: str = "cmr_subject_paths.pkl"
    target_tabular_data: str = "raw_tab.csv"
    
    augment: bool = True
    ignore_phenotype_tabular: bool = False # Whether the images have a row in tabular data is ignored
    
    num_train: int = 6000
    train_num_per_epoch: int = None
    num_val: int = 100
    num_test: int = 100
    batch_size: int = 1
    num_workers: int = 4
    
    dataset_cls: str = "Cardiac3DplusTAllAX"
    load_seg: bool = False
    all_value_names: Tuple[str, ...] = ("Age", "LVEDV (mL)", "LVESV (mL)", "LVSV (mL)", "LVEF (%)", "LVCO (L/min)", "LVM (g)", "RVEDV (mL)", "RVESV (mL)", "RVSV (mL)", "RVEF (%)", "LAV max (mL)", "LAV min (mL)", "LASV (mL)", "LAEF (%)", "RAV max (mL)", "RAV min (mL)", "RASV (mL)", "RAEF (%)")
    target_value_name: str = "LVM (g)" # Only select the ones that are involved in training 
    
    sax_slice_num: int = 6
    time_frame: int = 50
    image_size: Tuple[int, ...] = (128, 128)
    t_downsampling_ratio: int = 1
    frame_to_keep: int = 50
    
    # For visualization
    vis_token_path: str = "latent_code.npz"
    vis_tsne_map_path: str = "tsne.npz"
    save_all_patch_tokens: bool = True
    save_tsne: bool = True
    

@dataclass
class TrainerParams:
    accelerator: str = "gpu"
    max_epochs: int = 10_000
    check_val_every_n_epoch: int = 5
    devices: int = 1


@dataclass
class ReconMAEParams:
    enc_embed_dim: int = 1025 # has to be divisible by 8 or 6 for one modality. When it"s for two modalities, add one more dimension for distinguishing two modalities.
    enc_depth: int = 6
    enc_num_heads: int = 5
    mlp_ratio: float = 4.
    dec_embed_dim: int = 1025 # has to be divisible by 8 or 6 for one modality. When it"s for two modalities, add one more dimension for distinguishing two modalities.
    dec_depth: int = 2
    dec_num_heads: int = 5
    

@dataclass
class SegMAEParams:
    enc_embed_dim: int = 1025
    enc_depth: int = 6
    enc_num_heads: int = 5
    mlp_ratio: float = 4.
    feature_size: int = 16
    dec_embed_dim: int = 1152
    spatial_dims: int = 3
    upsample_kernel_sizes: Tuple[Tuple, ...] = ([1, 2, 2], [5, 2, 2], [5, 2, 2])
    

@dataclass
class RegrMAEParams:
    enc_embed_dim: int = 1025
    enc_depth: int = 6
    enc_num_heads: int = 5
    mlp_ratio: float = 4.
    dec_embed_dim: int = 256
    dec_depth: int = 2
    regressor_type: str = "cls_token"
    selected_features: list = None
    

@dataclass
class TrainingParams:
    test_scores_path: str = None
    grad_checkpointing: bool = False
    train_log_rate: int = 5
    val_log_rate: int = 10
    test_results_path: str = None
    test_psnr_path: str = None
    test_sample_path: str = None
    
    # Patchify
    patch_embed_cls: str = "PatchEmbed"
    patch_size: Tuple[int, ...] = (5, 8, 8)
    patch_in_channels: int = 1
    pixel_unshuffle_scale: int = 1
    mask_type: str = "random"
    mask_ratio: float = 0.7
    circular_pe: bool = False
    use_enc_pe: bool = True
    mask_loss: bool = True
    shift_size: Tuple[int] = (0, 0, 0)
    
    # Optimizer and scheduler
    dropout: float = 0.0
    lr: float = 1e-4
    min_lr: float = 0.0
    warmup_epochs: int = 20
    
    # Loss
    loss_types: Tuple[str, ...] = ("mse")
    loss_weights: Tuple[float, ...] = (1.0,)
    
    
@dataclass
class ModuleParams:
    task_idx: Optional[int] = None
    module_idx: Optional[int] = None
    
    training_hparams: TrainingParams = None
    recon_hparams: ReconMAEParams = None
    seg_hparams: SegMAEParams = None
    regr_hparams: RegrMAEParams = None
    
    
@dataclass
class Params:
    general: GeneralParams = None
    data: DataParams = None
    trainer: TrainerParams = None
    module: ModuleParams = None
