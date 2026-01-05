import os
from pathlib import Path
import pickle
import time
from uu import Error
import numpy as np
from torch.utils.data import random_split
from uu import Error
import tqdm
from typing import List, Union, Optional, Callable, Tuple
import nibabel as nib


# --- Envorinments variables --- #
IMAGE_SIZE = Path(os.environ["IMAGE_SIZE"])               # Desired image crop size (e.g., [128,128])
MAX_NUM_TRAIN = Path(os.environ["MAX_NUM_TRAIN"])         # Maximum number of training subjects
MAX_NUM_VAL = Path(os.environ["MAX_NUM_VAL"])             # Maximum number of validation subjects
MAX_NUM_TEST = Path(os.environ["MAX_NUM_TEST"])           # Maximum number of test subjects

IMAGE_DATA_ROOT = Path(os.environ["IMAGE_DATA_ROOT"])     # Root folder containing raw NIfTI images
IMAGE_PROCESS_ROOT = Path(os.environ["IMAGE_PROCESS_ROOT"]) # Folder to save preprocessed .npz files
IMAGE_SUBJ_PATH = Path(os.environ["IMAGE_SUBJ_PATH"]) # Path to pickle file storing train/val/test image paths
SELECTED_IDS_PICKLE = Path(os.environ["SELECTED_IDS_PICKLE"]) # Path to pickle file storing selected subject IDs


# --- Utility Functions --- #
def get_2D_slice_bbox_with_fixed_size(segmentation: np.ndarray, bbox_size: List[int] = [128, 128]) \
        -> np.ndarray:  # [Y1, X1, Y2, X2]
    """Find the biggest 2D slice and then crop or padding it into descired bbox_size"""
    if len(segmentation.shape) > 2:
        non_slice_dims = tuple(range(2, len(segmentation.shape)))
        segmentation = segmentation.max(axis=non_slice_dims)
    indices = np.argwhere(segmentation > 0)
    center_indices = (indices.min(0) + indices.max(0)) // 2
    min_indices = center_indices - np.array(bbox_size) // 2
    min_indices = np.maximum(min_indices, 0)
    max_indices = min_indices + np.array(bbox_size)
    max_indices = np.minimum(max_indices, segmentation.shape[:2])
    min_indices = max_indices - np.array(bbox_size)
    bbox = np.concatenate((min_indices, max_indices), axis=0)
    return bbox


def get_2D_central_bbox(im_size, bbox_size: List[int] = [128, 128]) \
        -> np.ndarray:  # [Y1, X1, Y2, X2]
    """Get the central bbox of the image with bbox_size"""
    center = np.array(im_size) // 2
    min_indices = center - np.array(bbox_size) // 2
    min_indices = np.maximum(min_indices, 0)
    max_indices = min_indices + np.array(bbox_size)
    max_indices = np.minimum(max_indices, im_size)
    min_indices = max_indices - np.array(bbox_size)
    bbox = np.concatenate((min_indices, max_indices), axis=0)
    return bbox


# --- Step 1: Find Subject IDs --- #
# Search through the raw image directory and collect IDs of subjects that meet:
# - Minimum number of SAX slices
# - Correct number of frames
# - Proper resolution
def find_indices_of_images(load_dir: Union[str, Path],
                           sax_file_name="sa.nii.gz",
                           seg_file_name="seg_sa.nii.gz",
                           lax_file_name=["la_2ch.nii.gz", "la_3ch.nii.gz", "la_4ch.nii.gz"],
                           sax_bbox_func: Optional[Callable] = get_2D_slice_bbox_with_fixed_size,
                           lax_bbox_func: Optional[Callable] = get_2D_central_bbox,
                           sax_bbox_size: Tuple[int, int] = [128, 128],
                           lax_bbox_size: Tuple[int, int] = [128, 128],
                           id_list: Optional[list] = None,):
    """
    Return list of subject IDs satisfying SAX/LAX image quality requirements.
    If id_list is given, only subjects in id_list are considered.
    """
    indices = []
    load_dir = Path(load_dir)
    for i, parent in enumerate(sorted(os.listdir(str(load_dir)))):
        if id_list is not None and int(parent) not in id_list:
            continue

        im_path = load_dir / parent / sax_file_name
        seg_path = load_dir / parent / seg_file_name
        lax_path = [load_dir / parent / i for i in lax_file_name]

        # Make sure both image and segmentation files exist
        if not os.path.exists(im_path) or not os.path.exists(seg_path) or not all(os.path.exists(p) for p in lax_path):
            continue

        try:
            im = nib.load(im_path)
            # Check slice and frame count
            if im.shape[2] < 6 or im.shape[3] != 50:
                print(f"Found an image with suspiciously low number of SAX slices/frames: {im.shape[2]} slices. {im.shape[3]} frames. {im_path.parent.name}")
                continue
            if sax_bbox_func and (im.shape[0] < sax_bbox_size[0] or im.shape[1] < sax_bbox_size[1]):
                print(f"Found an SAX image with suspiciously low resolution: {im.shape[0:2]} pixels. {im_path.parent.name}")
                continue
            lax_ims = [nib.load(p) for p in lax_path]
            if not all(l.shape[3] == 50 for l in lax_ims):
                print(f"Found an image with suspicious number of LAX frames: {lax_ims[0].shape[3], lax_ims[1].shape[3], lax_ims[3].shape[3]} frames. {im_path.parent.name}")
                continue
            if lax_bbox_func and not all(l.shape[0] >= lax_bbox_size[0] and l.shape[1] >= lax_bbox_size[1] for l in lax_ims):
                print(f"Found an LAX image with suspiciously low resolution: {lax_ims[0].shape[0:2], lax_ims[1].shape[0:2], lax_ims[2].shape[0:2]} pixels. {im_path.parent.name}")
                continue
            indices.append(int(parent))
        except Exception:
            continue
    return indices



# --- Step 2: Preprocess Selected Subjects into .npz --- #
def process_cmr_images(load_dir: Union[str, Path],
                       prep_dir: Union[str, Path],
                       num_cases: int = -1,
                       case_start_idx: int = 0,
                       sax_file_name="sa.nii.gz",
                       lax_file_name=["la_2ch.nii.gz", "la_3ch.nii.gz", "la_4ch.nii.gz"],
                       seg_sax_file_name="seg_sa.nii.gz",
                       seg_lax_file_name=["seg_la_2ch.nii.gz", None, "seg_la_4ch.nii.gz"],
                       file_name="processed_data.npy",
                       sax_bbox_func: Optional[Callable] = get_2D_slice_bbox_with_fixed_size,
                       lax_bbox_func: Optional[Callable] = get_2D_central_bbox,
                       sax_bbox_size: Tuple[int, int] = [128, 128],
                       lax_bbox_size: Tuple[int, int] = [128, 128],
                       id_list: Optional[list] = None,
                       replace_processed: Optional[bool] = False,):
    """
    Preprocess the raw NIfTI images into .npz arrays for selected subjects.
    Crops SAX/LAX images to bounding boxes, stacks LAX views, saves .npz per subject.
    """
    assert num_cases != 0
    assert os.path.exists(prep_dir), f"Processed directory {prep_dir} does not exist."
    count = 0
    # processed_npy_paths = []
    processed_case_ids = []
    start_time = time.time()
    load_dir = Path(load_dir)
    prep_dir = Path(prep_dir)
    
    dir_id_list = sorted(os.listdir(str(load_dir)))
    for i, parent in tqdm.tqdm(enumerate(dir_id_list), total=len(dir_id_list), desc="index list", unit=" iter", position=1):
        if i < case_start_idx:
            continue                            # Skip all subjects not belonging to this dataset
        if count >= num_cases > 0:
            break                               # If we have collected enough subjects, break
        if int(parent) not in id_list:
            continue                            # Skip all subjects not in the image dataset

        processed_npy_path = prep_dir / parent / file_name
        try:
            if replace_processed:
                raise Error("Reprocess the image data")
            if os.path.exists(processed_npy_path):
                process_npy = np.load(processed_npy_path)
                sax_im_data = process_npy["sax"].astype(np.float32) # [H, W, S, T]
                lax_im_data = process_npy["lax"].astype(np.float32) # [H, W, S, T]
                seg_sax_data = process_npy["seg_sax"].astype(np.int32) # [H, W, S, T]
                seg_lax_data = process_npy["seg_lax"].astype(np.int32)
                if sax_im_data.shape[:2] != (128, 128) or lax_im_data.shape[:2] != (128, 128)\
                    or seg_sax_data.shape[:2] != (128, 128) or seg_lax_data.shape[:2] != (128, 128):
                    raise Error("The size of the preprocessed npy data is incorrect. Reprocess the image data!")

                processed_case_ids.append(int(parent))
                count += 1
                continue                            # Skip all subjects that have already been processed
        except:
            sax_path = load_dir / parent / sax_file_name 
            lax_path = [load_dir / parent / i for i in lax_file_name]
            sax_seg_path = load_dir / parent / seg_sax_file_name
            lax_seg_path = [load_dir / parent / i if i is not None else None for i in seg_lax_file_name]
            
            if sax_bbox_func is not None:
                # Get bounding box of where foreground mask is present
                seg = nib.load(sax_seg_path).get_fdata()
                sax_bbox = sax_bbox_func(seg, sax_bbox_size)
            if lax_bbox_func is not None:
                lax_bbox = np.zeros((len(lax_path), 4), dtype=np.int32)
                for i in range(len(lax_path)):
                    lax_im = nib.load(lax_path[i]).get_fdata()
                    lax_bbox[i] = lax_bbox_func(lax_im.shape[:2], lax_bbox_size)
                
            # Load cropped sax images and segmentations into arrays
            nii_sax = nib.load(sax_path).get_fdata().astype(np.float32)
            nii_seg_sax = nib.load(sax_seg_path).get_fdata().astype(np.int32)
            raw_shape = nii_sax.shape
            assert len(sax_bbox) % 2 == 0
            if len(sax_bbox) // 2 == 2:
                sax_bbox = (*sax_bbox[:2], 0, *sax_bbox[-2:], raw_shape[2])
            if len(sax_bbox) // 2 == 3:
                sax_bbox = (*sax_bbox[:3], 0, *sax_bbox[-3:], sax_bbox[3])
            idx_slices = (slice(sax_bbox[0], sax_bbox[0 + len(sax_bbox)//2]),
                        slice(sax_bbox[1], sax_bbox[1 + len(sax_bbox)//2]),
                        )
            sax_arrs = nii_sax[idx_slices]
            seg_sax_arrs = nii_seg_sax[idx_slices]
            
            # Load cropped long images into an array
            laxs = []
            seg_laxs = []
            for i in range(len(lax_path)):
                nii_lax = nib.load(lax_path[i]).get_fdata().astype(np.float32)
                raw_shape = nii_lax.shape
                assert len(lax_bbox[i]) % 2 == 0
                if len(lax_bbox[i]) // 2 == 2:
                    lax_bbox_ = (*lax_bbox[i][:2], 0, *lax_bbox[i][-2:], raw_shape[2])
                if len(lax_bbox[i]) // 2 == 3:
                    lax_bbox_ = (*lax_bbox[i][:3], 0, *lax_bbox[i][-3:], lax_bbox[i][3])
                idx_slices = (slice(lax_bbox_[0], lax_bbox_[0 + len(lax_bbox_)//2]),
                            slice(lax_bbox_[1], lax_bbox_[1 + len(lax_bbox_)//2]),
                            )
                lax = nii_lax[idx_slices]
                laxs.append(lax)

                if lax_seg_path[i] is not None:
                    nii_seg_lax = nib.load(lax_seg_path[i]).get_fdata().astype(np.int32)
                    seg_lax = nii_seg_lax[idx_slices]
                    seg_laxs.append(seg_lax)
                else:
                    pad_seg_lax = np.zeros_like(lax).astype(np.int32)
                    seg_laxs.append(pad_seg_lax)

            lax_arrs = np.stack(laxs, axis=2).squeeze(-2)
            seg_lax_arrs = np.stack(seg_laxs, axis=2).squeeze(-2)
            
            # Save to a numpy file
            if not os.path.exists(processed_npy_path.parent):
                os.makedirs(processed_npy_path.parent)
            if replace_processed and os.path.exists(processed_npy_path):
                os.remove(processed_npy_path)
            processed_npy = {"sax": sax_arrs, "lax": lax_arrs, 
                             "seg_sax": seg_sax_arrs, "seg_lax": seg_lax_arrs}
            if file_name[-4:] == ".npy":
                np.save(processed_npy_path, processed_npy)
            elif file_name[-4:] == ".npz":
                np.savez(processed_npy_path, sax=sax_arrs, lax=lax_arrs, seg_sax=seg_sax_arrs, seg_lax=seg_lax_arrs)
            else:
                raise NotImplementedError
            processed_case_ids.append(int(parent))
            
            count += 1
    print(f"{count} cases are found")
    if num_cases > 0 and count != num_cases:
        raise ValueError(f"Did not find required amount of cases ({num_cases}) in directory: {load_dir}")
    
    elapsed = time.time() - start_time
    print(f"Processed {count} cases in out of {len(dir_id_list)} in {elapsed//60}m {int(elapsed%60)}s.")
    print(f"The data searching range is from {case_start_idx} to {i}.")
    return processed_case_ids


# --- Step 3: Create Pickle File with All .npz Paths --- #
def process_image_and_save_paths(path_dir: str, 
                                 subj_ids_dir: str, 
                                 load_dir: str, 
                                 processed_dir: str,
                                 processed_file_name: str, 
                                 num_train: int, 
                                 num_val: int, 
                                 num_test: int,
                                 idx_start: int,
                                 image_size: list[int] = [128, 128],
                                 replace_processed: bool = False,
                                 ):
    """
    - Load or compute selected subject IDs
    - Preprocess raw NIfTI to .npz for selected IDs
    - Split processed .npz into train/val/test
    - Save paths dictionary as pickle
    """
    if os.path.exists(subj_ids_dir):
        with open(subj_ids_dir, 'rb') as file:
            subject_ids = pickle.load(file)
    else:
        subj_ids = find_indices_of_images(load_dir, sax_bbox_size=image_size, lax_bbox_size=image_size)
        subject_ids = process_cmr_images(load_dir=load_dir, 
                                         prep_dir=processed_dir, 
                                         file_name=processed_file_name,
                                         sax_bbox_size=image_size,
                                         lax_bbox_size=image_size, 
                                         replace_processed=replace_processed,
                                         id_list=subj_ids)
        subject_ids = sorted(subject_ids)
        with open(subj_ids_dir, 'wb') as file:
            pickle.dump(subject_ids, file)

    # Split into train/val/test paths
    subject_paths = [Path(processed_dir) / str(parent) / Path(processed_file_name) for parent in subject_ids]
    split = (num_train, num_val, num_test)
    topk_paths = subject_paths[idx_start : (sum(split) + idx_start)]
    train_idxs, val_idxs, test_idxs = [list(s) for s in random_split(list(range(len(topk_paths))), split)]
    paths = {
        "train": [topk_paths[i] for i in train_idxs],
        "val": [topk_paths[i] for i in val_idxs],
        "test": [topk_paths[i] for i in test_idxs]
    }
    with open(path_dir, 'wb') as handle:
        pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return paths



if __name__ == "__main__":
    # --- Configurations --- #
    replace_processed = False
    
    # --- Main Execution --- #
    # Load existing pickle with train/val/test .npz paths or preprocess everything if missing/replaced
    try:
        if replace_processed:
            raise Error("Reprocess the image data")
    
        with open(IMAGE_SUBJ_PATH, 'rb') as file:
            paths = pickle.load(file)
        imgs_n = len(paths["train"]) + len(paths["val"]) + len(paths["test"])
        print(f"Loaded {imgs_n} images from {IMAGE_SUBJ_PATH}.")

    except:
        paths = process_image_and_save_paths(path_dir=IMAGE_SUBJ_PATH, 
                                            subj_ids_dir=SELECTED_IDS_PICKLE, 
                                            load_dir=IMAGE_DATA_ROOT, 
                                            processed_dir=IMAGE_PROCESS_ROOT,
                                            processed_file_name="processed_seg_allax.npz", 
                                            num_train=MAX_NUM_TRAIN, 
                                            num_val=MAX_NUM_VAL, 
                                            num_test=MAX_NUM_TEST,
                                            idx_start=0,
                                            image_size=IMAGE_SIZE,
                                            replace_processed=replace_processed)
