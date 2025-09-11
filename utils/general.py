import argparse
import numpy as np
import torch
import os
from typing import Union
import dataclasses
import shutil
from skimage.metrics import hausdorff_distance as skhausdorff
from huggingface_hub import hf_hub_download, list_repo_files


def parser_command_line():
    "Define the arguments required for the script"
    parser = argparse.ArgumentParser(description="Masked Autoencoder Downstream Tasks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest="pipeline", help="pipeline to run")
    
    # Arguments for training
    parser_train = subparser.add_parser("train", help="train the imaging model")
    parser_train.add_argument("-c", "--config", help="config file (.yml) containing the ¢hyper-parameters for inference.")
    parser_train.add_argument("-m", "--module", default="imaging", help="imaging or imaging_tabular.")
    parser_train.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_train.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    
    # Arguments for validation
    parser_eval = subparser.add_parser("val", help="validate the model")
    parser_eval.add_argument("-c", "--config", help="config file (.yml) containing the hyper-parameters for inference.")
    parser_eval.add_argument("-m", "--module", default="imaging", help="imaging or imaging_tabular.")
    parser_eval.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_eval.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    
    # Arguments for testing
    parser_test = subparser.add_parser("test", help="test  the model")
    parser_test.add_argument("-c", "--config", help="config file (.yml) containing the hyper-parameters for inference.")
    parser_test.add_argument("-m", "--module", default="imaging", help="imaging or imaging_tabular.")
    parser_test.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_test.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")

    return parser.parse_args()


@dataclasses.dataclass
class PathHolder:
    dataloader_image_file_folder: str 
    dataloader_tabular_file_folder: str
    image_subject_paths_folder: str
    raw_tabular_data_path: str
    input_tabular_data_path: str
    log_folder: str


def get_data_paths():
    return PathHolder(
        dataloader_image_file_folder=os.path.join(os.environ["DATALOADER_IMAGE_FILE_ROOT"]),
        dataloader_tabular_file_folder=os.path.join(os.environ["DATALOADER_TABULAR_FILE_ROOT"]),
        image_subject_paths_folder=os.path.join(os.environ["IMAGE_SUBJ_PATHS_FOLDER"]),
        raw_tabular_data_path=os.path.join(os.environ["RAW_TABULAR_DATA_PATH"]),
        input_tabular_data_path=os.path.join(os.environ["PREPROCESSED_TABULAR_DATA_PATH"]),
        log_folder=os.path.join(os.environ["LOG_FOLDER"]),
        )


def download_checkpoints(repo_id="UKBB-Foundational-Models/ViTa",
                         log_dir="./log",
                         cache_dir="./hf_cache"):
    """
    Download checkpoints from a Hugging Face repo and organize them into folders.
    Skips downloading if the checkpoint already exists in the target folder.
    """
    # Define target directories
    ckpt_imaging = os.path.join(log_dir, "checkpoints_imaging")
    ckpt_imaging_tabular = os.path.join(log_dir, "checkpoints_imaging_tabular")

    os.makedirs(ckpt_imaging, exist_ok=True)
    os.makedirs(ckpt_imaging_tabular, exist_ok=True)

    # List files in repo
    ckpt_files = [
        "downstream_clas_vita_cad.ckpt", 
        "downstream_clas_vita_diabetes.ckpt", 
        "downstream_clas_vita_high_blood_pressure.ckpt",
        "downstream_clas_vita_hypertension.ckpt",
        "downstream_clas_vita_infarct.ckpt",
        "downstream_clas_vita_stroke.ckpt",
        "downstream_pred_vita_agewhenattendedassessmentcentre.ckpt",
        "downstream_pred_vita_allindicators.ckpt",
        "downstream_pred_vita_allphenotypes_lax.ckpt",
        "downstream_pred_vita_allphenotypes_sax.ckpt",
        "downstream_seg_mae_allax.ckpt",
        "pretrain_mae_allax.ckpt",
        "pretrain_vita.ckpt"
        ]

    # print(f"Found {len(ckpt_files)} checkpoint files in {repo_id}:")
    # for f in ckpt_files:
    #     print(" -", f)

    saved_paths = {"imaging": [], "imaging_tabular": []}

    # Download & save
    for f in ckpt_files:
        # Decide target path
        if "vita" in f.lower():
            target_path = os.path.join(ckpt_imaging_tabular, os.path.basename(f))
            saved_paths["imaging_tabular"].append(target_path)
        else:
            target_path = os.path.join(ckpt_imaging, os.path.basename(f))
            saved_paths["imaging"].append(target_path)

        # Skip if already exists
        if os.path.exists(target_path):
            print(f"⏩ Skipping {f}, already exists at {target_path}")
            continue

        # Otherwise download and copy
        local_path = hf_hub_download(repo_id=repo_id, filename=f, cache_dir=cache_dir)
        print(f"⬇️  Downloading {f} → {target_path}")
        shutil.copy(local_path, target_path)

    print("✅ Done! All checkpoints are available locally.")
    return saved_paths


def normalize_image(im: Union[np.ndarray, torch.Tensor], low: float = None, high: float = None, clip=True, 
                    scale: float=None) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize array to range [0, 1] """
    if low is None:
        low = im.min()
    if high is None:
        high = im.max()
    if clip:
        im = im.clip(low, high)
    im_ = (im - low) / (high - low)
    if scale is not None:
        im_ = im_ * scale
    return im_


def image_normalization(image, scale=1, mode="2D"):
    if isinstance(image, np.ndarray) and np.iscomplexobj(image):
        image = np.abs(image)
    low = image.min()
    high = image.max()
    im_ = (image - low) / (high - low)
    if scale is not None:
        im_ = im_ * scale
    return im_


def to_1hot(class_indices: torch.Tensor, num_class) -> torch.Tensor:
    """ Converts index array to 1-hot structure. """
    origin_shape = class_indices.shape
    class_indices_ = class_indices.view(-1, 1).squeeze(1)
    
    N = class_indices_.shape[0]
    seg = class_indices_.to(torch.long).reshape((-1,))
    seg_1hot_ = torch.zeros((N, num_class), dtype=torch.float32, device=class_indices_.device)
    seg_1hot_[torch.arange(0, seg.shape[0], dtype=torch.long), seg] = 1
    seg_1hot = seg_1hot_.reshape(*origin_shape, num_class)
    return seg_1hot


def hausdorff_distance(pred: torch.Tensor, gt: torch.Tensor):
    """
    Calculates the Hausdorff distance. pred is in the shape of (C, T H, W), and gt is in the shape of (C, T, H, W). T is the number of time frames, C is the number of classes, H is the height, and W is the width.
    """
    assert pred.shape == gt.shape, "The two sets must have the same shape."
    hd = torch.empty(pred.shape[:2])
    for c in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            hd[c, t] = skhausdorff(pred[c][t].detach().cpu().numpy(), gt[c][t].cpu().numpy())
    return hd.mean(dim=1)