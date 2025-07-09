import wandb
import numpy as np
import torch
from utils.general import normalize_image
from PIL import Image
 

class CustomWandbLogger:
    def __init__(self, vis_log_video_fps=2,):
        self.log_train_metric_dict = {}
        self.log_val_metric_dict = {}
        self.log_test_metric_dict = {}
        self.log_train_img_dict = {}
        self.log_train_video_dict = {}
        self.log_val_img_dict = {}
        self.log_val_video_dict = {}
        self.log_test_img_dict = {}
        self.log_test_video_dict = {}
        self.vis_log_video_fps = vis_log_video_fps
    
    def update_metric_item(self, item, value, mode="train"):
        if item not in eval(f"self.log_{mode}_metric_dict"):
            eval(f"self.log_{mode}_metric_dict")[item] = [value]
        else:
            eval(f"self.log_{mode}_metric_dict")[item].append(value)

    def update_img_item(self, vis_item, subj_name, value, mode="train"):
        if vis_item not in eval(f"self.log_{mode}_img_dict"):
            eval(f"self.log_{mode}_img_dict")[vis_item] = {}
        eval(f"self.log_{mode}_img_dict")[vis_item][subj_name] = wandb.Image(value, caption=subj_name)

    def update_video_item(self, vis_item, subj_name, value, mode="train"):
        if vis_item not in eval(f"self.log_{mode}_video_dict"):
            eval(f"self.log_{mode}_video_dict")[vis_item] = {}
        eval(f"self.log_{mode}_video_dict")[vis_item][subj_name] =wandb.Video(value, caption=subj_name, fps=self.vis_log_video_fps)
            
    def reset_item(self):
        self.log_train_metric_dict = {key: [] for key in self.log_train_metric_dict}
        self.log_val_metric_dict = {key: [] for key in self.log_val_metric_dict}
        self.log_train_img_dict = {key: {} for key in self.log_train_img_dict}
        self.log_val_img_dict = {key: {} for key in self.log_val_img_dict}
        self.log_train_video_dict = {key: {} for key in self.log_train_video_dict}
        self.log_val_video_dict = {key: {} for key in self.log_val_video_dict}
    
    def reset_test_item(self):
        self.log_test_metric_dict = {key: [] for key in self.log_test_metric_dict}
        self.log_test_img_dict = {key: {} for key in self.log_test_img_dict}
        self.log_test_video_dict = {key: {} for key in self.log_test_video_dict}


def fix_dict_in_wandb_config(wandb):
    """"Adapted from [https://github.com/wandb/client/issues/982]"""
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if "." in k:
            keys = k.split(".")
            if len(keys) == 2:
                new_key = k.split(".")[0]
                inner_key = k.split(".")[1]
                if new_key not in config.keys():
                    config[new_key] = {}
                config[new_key].update({inner_key: v})
                del config[k]
            elif len(keys) == 3:
                new_key_1 = k.split(".")[0]
                new_key_2 = k.split(".")[1]
                inner_key = k.split(".")[2]

                if new_key_1 not in config.keys():
                    config[new_key_1] = {}
                if new_key_2 not in config[new_key_1].keys():
                    config[new_key_1][new_key_2] = {}
                config[new_key_1][new_key_2].update({inner_key: v})
                del config[k]
            else: # len(keys) > 3
                raise ValueError("Nested dicts with depth>3 are currently not supported!")

    wandb.config = wandb.Config()
    for k, v in config.items():
        wandb.config[k] = v
        

def pick_imgs_for_wandb(imgs):
    # If there are multiple slices, take two middle slices.
    # Otherwise, take the first slice and the last slice.
    if imgs.shape[0] > 1:
        if imgs.shape[0] == 9:
            video = torch.cat([imgs[0], imgs[1],imgs[2], imgs[5], imgs[7]], dim=2)
        elif imgs.shape[0] == 3:
            video = torch.cat([imgs[0], imgs[1], imgs[2]], dim=2)
        else:
            video = torch.cat([imgs[i] for i in range(imgs.shape[0])], dim=2)
    else:
        video = imgs[0]
    return video


def imgs_to_wandb_video(imgs, scale=1, prep=False, in_channel=1):
    """Process 3D+t images to wandb video format
    imgs: [T, H, W] torch.Tensor or np.ndarray
    prep: bool. Whether the input images are preprocess for logging.
    video: [T, 3, H, W * 2] or [T, 3, H, W] np.ndarray
    """
    if prep:
        video = imgs
    else:
        video = pick_imgs_for_wandb(imgs)
    video_log = video[:, None, ...] if in_channel == 1 else video
    if scale != 1:
        video_log = normalize_image(video_log)
    video_log = video_log * 255.
    if isinstance(video_log, torch.Tensor):
        if video_log.device.type == "cuda":
            video_log = video_log.detach().cpu()
        video_log = video_log.numpy()
    video_log = video_log.astype(np.uint8)
    
    # Reshape to [T, 3, H, W]
    video_log = video_log.repeat(3, axis=1) if in_channel == 1 else video_log
    return video_log


def imgs_to_wandb_image(imgs):
    """Process 3D+t images to wandb image format
    imgs: [S, T, H, W] torch.Tensor or np.ndarray
    image: [3, H, W]
    """
    # Take the first slice of the 3D volume
    image_log = imgs[0, 0, ..., None] * 255
    if isinstance(image_log, torch.Tensor):
        if image_log.device.type == "cuda":
            image_log = image_log.detach().cpu()
        image_log = image_log.numpy()
    image_log = image_log.astype(np.uint8)
    
    # Reshape to [3, H, W]
    image_log = image_log.repeat(3, axis=-1)
    return image_log


def imgs_to_wandb_image_3D(imgs):
    """Process 3D images to wandb image format
    imgs: [S, H, W] torch.Tensor or np.ndarray
    image: [3, H, W]
    """
    # Take the first slice of the 3D volume
    if isinstance(imgs, torch.Tensor):
        if imgs.device.type == "cuda":
            imgs = imgs.detach().cpu()
    imgs_list = []
    for i in range(imgs.shape[0]):
        imgs_list += [imgs[i]]
    imgs_cat = torch.concat(imgs_list, dim=1).numpy()
    image_log = imgs_cat[..., None] * 255
    
    # Reshape to [3, H, W]
    image_log = image_log.astype(np.uint8).repeat(3, axis=-1)
    return image_log


def replace_with_gt_wandb_video(pred, img, mask):
    """Replace the prediction with the ground truth based on mask in the wandb video format
    pred: [T, 3, H, W] or [T, H, W] np.ndarray
    img: [T, 3, H, W] or [T, H, W] np.ndarray
    mask: [T, 3, H, W] or [T, H, W] np.ndarray"""
    assert pred.shape == img.shape == mask.shape
    if len(pred.shape) == 4:
        pred = pred[:, 0, ...]
        img = img[:, 0, ...]
        mask = mask[:, 0, ...]
    replace_pred = np.where(mask == 0, img, pred)
    replace_pred = replace_pred[:, None, ...].repeat(3, axis=1)
    return replace_pred


def overlay_segmentation(grayscale_array, segmentation_array, modality):                    
    if modality == "sax":
        # Define your colors for each class, including an alpha channel for transparency
        colors = [
            (0, 0, 0, 0),        # Class 0 color (Black with zero transparency)
            (255, 0, 0, 128),    # Class 1 color (Red with half transparency)
            (0, 255, 0, 128),    # Class 2 color (Green with half transparency)
            (0, 0, 255, 128),    # Class 3 color (Blue with half transparency)
        ]
    elif modality == "lax":
        colors = [
            (0, 0, 0, 0),        # Class 0 color (Black with zero transparency)
            (255, 255, 0, 128),  # Class 4 color (Yellow with half transparency)
            (255, 0, 255, 128)   # Class 5 color (Magenta with half transparency)
        ]
    # Ensure grayscale_array is a 2D array [H, W] and range to 255.
    if grayscale_array.ndim != 2:
        raise ValueError("The grayscale_array should be a 2D array")
    if grayscale_array.max() <= 1.0:
        grayscale_array = grayscale_array * 255.
        
    # Convert the grayscale image to RGBA
    height, width = grayscale_array.shape
    grayscale_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    grayscale_rgba[..., :3] = grayscale_array[..., np.newaxis]  # Assign the same value to R, G, and B
    grayscale_rgba[..., 3] = 255  # Full alpha
    
    # Convert the grayscale RGBA array to an Image
    original_img = Image.fromarray(grayscale_rgba, "RGBA")
    
    # Prepare an empty image with the same size as the original image and RGBA channels
    segmentation_img = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Assign colors to the segmentation image based on the one-hot encoded classes
    for class_index, color in enumerate(colors):
        segmentation_mask = segmentation_array[:, :, class_index]
        segmentation_img[segmentation_mask == 1] = color  # Set the color for the current class
    
    # Convert the RGBA segmentation array to an Image
    segmentation_img_pil = Image.fromarray(segmentation_img, "RGBA")
    
    # Overlay the segmentation on the original image
    combined_image = Image.alpha_composite(original_img, segmentation_img_pil)
    # Image.blend(original_img, segmentation_img_pil, alpha)
    return combined_image


def delete_media(entity, project, run_id):
    """Delete the images and videos in one job on wandb."""

    import wandb
    api = wandb.Api()

    run = api.run(f"{entity}/{project}/{run_id}")

    extension_list = [".png", "gif"]
    files = run.files()
    for file in files:
        for extension in extension_list:
            if file.name.endswith(extension):
                file.delete()
    return