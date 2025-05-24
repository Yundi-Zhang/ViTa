import argparse
import numpy as np
import torch
import os
from typing import Optional, Union
import dataclasses
import socket


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
    parser_eval = subparser.add_parser("test", help="test  the model")
    parser_eval.add_argument("-c", "--config", help="config file (.yml) containing the hyper-parameters for inference.")
    parser_eval.add_argument("-m", "--module", default="imaging", help="imaging or imaging_tabular.")
    parser_eval.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_eval.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    
    # Arguments for evaluation
    parser_vis = subparser.add_parser("vis", help="visualize the results of the latent space")
    parser_vis.add_argument("-c", "--config", help="config file (.yml) containing the hyper-parameters for inference.")
    parser_vis.add_argument("-m", "--module", default="imaging", help="imaging or imaging_tabular.")
    parser_vis.add_argument("-g", "--wandb_group_name", default=None, help="specify the name of the group")
    parser_vis.add_argument("-n", "--wandb_run_name", default=None, help="specify the name of the experiment")
    return parser.parse_args()


@dataclasses.dataclass
class PathHolder:
    # Image data paths
    image_root_folder: str
    image_processed_folder: str
    
    # Tabular data paths
    all_feature_tabular_dir: str
    biomarker_tabular_dir: str
    # tabular_root_folder: str
    
    # Labels data root path
    # labels_root_folder: str
    
    dataloader_image_file_folder: str # Processed file path
    dataloader_tabular_file_folder: str # Processed file path
    log_folder: str # Logging path


def get_data_paths():
    return PathHolder(image_root_folder=os.path.join(os.environ["IMAGE_DATA_ROOT"]),
                      image_processed_folder=os.path.join(os.environ["IMAGE_PROCESS_ROOT"]),
                      all_feature_tabular_dir=os.path.join(os.environ["ALL_FEATURE_TABULAR_DIR"]),
                      biomarker_tabular_dir=os.path.join(os.environ["BIOMARKER_TABULAR_DIR"]),
                      log_folder=os.path.join(os.environ["LOG_FOLDER"]),
                      dataloader_image_file_folder=os.path.join(os.environ["DATALOADER_IMAGE_FILE_ROOT"]),
                      dataloader_tabular_file_folder=os.path.join(os.environ["DATALOADER_TABULAR_FILE_ROOT"]),
                      )
    
    
def get_computer_id():
    hostname = socket.gethostname()
    if hostname == "unicorn":
        return "yundi-wks"
    elif hostname in ['atlas', 'chameleon', 'helios', 'prometheus', 'leto', 'hercules', 'apollo']:  # GPU server
        logname = os.environ["LOGNAME"]
        return f"{logname}-gpu"
    else:
        raise Exception(f"Unknown hostname: {hostname}.")


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
