"""Parameters for the module in a nested dataclass manner."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GeneralParams:
    # wandb logging configs
    wandb_project_name: str = "ViTa"
    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_disabled: str = "true"
    
    seed: int = 1
    gpu: int = 0
    
    resume_training: bool = False
    resume_ckpt_path: str = None

    
@dataclass
class ImagingDataParams:
    cmr_path_pickle_name: str = "cmr_subject_paths.pkl"
    augment: bool = True
    ignore_phenotype_tabular: bool = False # Whether the images have a row in tabular data is ignored
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
    

@dataclass
class TabularDataParams:
    tabular_data: str = "clean_tab.csv"
    raw_tabular_data: str = "raw_tab_paul.csv"
    field_lengths_tabular: str = "tabular_lengths.pt"
    corruption_rate: int = 0.3
    replace_tabular_data: bool = False
    one_hot_tabular: bool = True
    normalize_tabular: bool = True
    tab_augment: bool = True


@dataclass
class ValidationDataParams:
    validation_form: str = "classification"
    selected_cols: Tuple[str, ...] = "Age"
    labels: str = "labels_CAD_all.pt"
    
    
@dataclass
class GeneralDataParams:
    dataset_cls: str = "Cardiac3DplusTAllAX"
    num_train: int = 6000
    train_num_per_epoch: int = None
    num_val: int = 100
    num_test: int = 100
    batch_size: int = 1
    num_workers: int = 4
    stratified_sampler: bool = False
    balanced_sampler: bool = False
    

@dataclass
class TrainerParams:
    accelerator: str = "gpu"
    max_epochs: int = 10_000
    check_val_every_n_epoch: int = 5
    devices: int = 1
    num_nodes: int = 1


@dataclass
class ImagingParams:
    load_imaging_encoder: bool = False
    freeze_imaging_encoder: bool = False
    load_imaging_decoder: bool = False
    freeze_imaging_decoder: bool = False
    imaging_ckpt_path: str = None
    
    # Patchify
    patch_embed_cls: str = "PatchEmbed"
    patch_size: Tuple[int, ...] = (5, 8, 8)
    patch_in_channels: int = 1
    pixel_unshuffle_scale: int = 1
    mask_type: str = "random"
    mask_ratio: float = 0.0
    circular_pe: bool = False
    use_enc_pe: bool = True
    mask_loss: bool = True
    shift_size: Tuple[int] = (0, 0, 0)
    
    # Network
    # Encoder
    enc_embed_dim: int = 1025 # has to be divisible by 8 or 6 for one modality. 
                            # When it"s for two modalities, add one more for distinguishing two modalities.
    enc_depth: int = 6
    enc_num_heads: int = 5
    mlp_ratio: float = 4.
    projection_dim: int = 256
    # Decoder
    dec_embed_dim: int = 1025 # has to be divisible by 8 or 6 for one modality. When it"s for two modalities, add one more dimension for distinguishing two modalities.
    dec_depth: int = 2
    dec_num_heads: int = 5
    mlp_ratio: float = 4.
    grad_checkpointing: bool = False

@dataclass
class TabularParams:
    tabular_encoder_type: str = "TabularMLP"
    load_tabular_encoder: bool = False
    freeze_tabular_encoder: bool = False
    tabular_ckpt_path: str = None
    
    input_size: int = 2
    embedding_dim: int = 1025 # = enc_embed_dim

    corruption_rate: float = 0.3
    one_hot: bool = True
    eval_one_hot: bool = True

    encoder_num_layers: int = 2
    encoder_num_heads: int = 2
    encoder_mlp_ratio: float = 4.
    projector_num_layers: int = 1
    init_strat: str = "kaiming"
    projection_dim: int = 256
    # Decoder
    decoder_dim: int = 1024
    decoder_out_dim: int = 1024
    decoder_depth: int = 2
    decoder_type: str = "linear"
    decoder_head_type: str = "numerical"
    use_scalor: bool = False
    selected_features: Tuple[str, ...] = None
    grad_checkpointing: bool = False
    pos_weight: Tuple[float] = (1.0)


@dataclass
class TrainingParams:
    avg_token: bool = True
    train_log_rate: int = 5
    val_log_rate: int = 10
    classifier_freq: int = 1
    log_images: bool = False
    
    # Optimizer and scheduler
    dropout: float = 0.0
    lr: float = 1e-4
    min_lr: float = 0.0
    
    # Loss
    loss_types: Tuple[str, ...] = ("mse")
    loss_weights: Tuple[float, ...] = (1.0,)
    
    temperature: float = 0.1
    lambda_0: float = 0.5
    num_classes: int = 2
    weight_decay: float = 1.e-4
    scheduler: str = "anneal"
    anneal_max_epochs: int = 200
    warmup_epochs: int = 10


@dataclass
class DataParams:
    general_data: GeneralDataParams = None
    imaging_data: ImagingDataParams = None
    tabular_data: TabularDataParams = None
    validation_data: ValidationDataParams = None
     
    
@dataclass
class ModuleParams:
    task_idx: Optional[int] = None
    module_idx: Optional[int] = None
    test_results_path: Optional[str] = None
    
    training_hparams: TrainingParams = None
    imaging_hparams: ImagingParams = None
    tabular_hparams: TabularParams = None
    
    
@dataclass
class Params:
    general: GeneralParams = None
    data: DataParams = None
    trainer: TrainerParams = None
    module: ModuleParams = None
