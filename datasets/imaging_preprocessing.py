import os
from pathlib import Path
import pickle
from typing import Optional
from uu import Error
import numpy as np
import pandas as pd
from torch.utils.data import random_split

from utils.imaging_data_related import find_healthy_subjects, find_indices_of_images, process_cmr_images


def load_image_paths(path_dir: str, subj_ids_dir: str, load_dir: str, processed_dir: str, 
                     replace_processed: bool = False, **kwargs,):
    try:
        if replace_processed:
            raise Error("Reprocess the image data")
        
        # Load image paths
        with open(path_dir, 'rb') as file:
            paths = pickle.load(file)
        imgs_n = len(paths["train"]) + len(paths["val"]) + len(paths["test"])
        print(f"Loaded {imgs_n} images from {path_dir}.")
    except:
        paths = process_image_and_save_paths(path_dir, subj_ids_dir, load_dir, processed_dir,
                                             processed_file_name=kwargs.get("processed_file_name"), 
                                             num_train=kwargs.get("num_train"), 
                                             num_val=kwargs.get("num_val"), 
                                             num_test=kwargs.get("num_test"),
                                             idx_start=kwargs.get("idx_start"),
                                             image_size=kwargs.get("image_size"),
                                             replace_processed=kwargs.get("replace_processed"),)
    return paths


def load_image_paths_and_target_table(path_dir: str,
                                     subj_ids_dir: str,
                                     load_dir: str,
                                     processed_dir: str, 
                                     replace_processed: bool = False,
                                     target_table_dir: Optional[str] = None,
                                     target_value_name: str = "Age",
                                     processed_table_pickle_dir: Optional[str] = None,
                                     **kwargs,):
    
    try:
        if replace_processed:
            raise Error("Reprocess the tabular data")
        
        # Load the table with target phenotypes
        with open(target_table_dir, 'rb') as file:
            target_table = pickle.load(file)
        
        # Load image paths
        if target_value_name in target_table.columns:
            with open(path_dir, 'rb') as handle:
                paths = pickle.load(handle)
            imgs_n = len(paths["train"]) + len(paths["val"])+len(paths["test"])
            print(f"Loaded {imgs_n} images from {path_dir}.")
        else:
            raise Error("The target value is not in the target table")
    except:
        paths, target_table = process_image_and_create_target_table(
            path_dir, subj_ids_dir, load_dir, processed_dir,
            processed_table_pickle_dir=processed_table_pickle_dir,
            biomarker_table_pickle_dir=target_table_dir,
            processed_file_name=kwargs.get("processed_file_name"), # Image
            all_feature_tabular_dir=kwargs.get("all_feature_tabular_dir"), # Tabular
            biomarker_tabular_dir=kwargs.get("biomarker_tabular_dir"), # Tabular
            table_condition_dict=kwargs.get("table_condition_dict"), # Tabular
            num_train=kwargs.get("num_train"), 
            num_val=kwargs.get("num_val"), 
            num_test=kwargs.get("num_test"),
            idx_start=kwargs.get("idx_start"),
            all_value_names=kwargs.get("all_value_names"),
            target_value_name=target_value_name,
            image_size=kwargs.get("image_size"),
            replace_processed=replace_processed,
            data_filtering=kwargs.get("data_filtering"),
            )
                                                                                
    return paths, target_table


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
    # Find the subjects with enough sax slice number and proper image size
    if os.path.exists(subj_ids_dir):
        with open(subj_ids_dir, 'rb') as file:
            subject_ids = pickle.load(file)
    else:
        subj_ids = find_indices_of_images(load_dir, sax_bbox_size=image_size, lax_bbox_size=image_size)
        # Process the images and segmentations into npz files if not preprocessed
        subject_ids = process_cmr_images(load_dir=load_dir, 
                                         prep_dir=processed_dir, 
                                         file_name=processed_file_name,
                                         sax_bbox_size=image_size,
                                         lax_bbox_size=image_size, 
                                         replace_processed=replace_processed,
                                         id_list=subj_ids) 
        subject_ids = sorted(subject_ids) # Subjects with CMR images
        with open(subj_ids_dir, 'wb') as file:
            pickle.dump(subject_ids, file)
    #####################################################        
    # Split training, validation, and test subject paths   
    subject_paths = []
    for parent in subject_ids:
        subject_path = Path(processed_dir) / str(parent) / Path(processed_file_name)
        subject_paths.append(subject_path)
            
    split = (num_train, num_val, num_test)
    topk_paths = subject_paths[idx_start : (sum(split) + idx_start)]
    train_idxs, val_idxs, test_idxs = [list(s) for s in random_split(list(range(len(topk_paths))), split)]
    train_paths = [topk_paths[i] for i in train_idxs]
    val_paths = [topk_paths[i] for i in val_idxs]
    test_paths = [topk_paths[i] for i in test_idxs]
    paths = {"train": train_paths, "val": val_paths, "test": test_paths}
    with open(path_dir, 'wb') as handle:
        pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return paths


def process_image_and_create_target_table(path_dir, 
                                          subj_ids_dir, 
                                          load_dir: str, 
                                          processed_dir: str,  # Image
                                          processed_table_pickle_dir: str,
                                          biomarker_table_pickle_dir: str,
                                          processed_file_name: str = "processed_seg_allax.npz", # Image
                                          all_feature_tabular_dir: Optional[str] = None, # Tabular
                                          biomarker_tabular_dir: Optional[str] = None, # Tabular
                                          table_condition_dict: Optional[dict] = None, # Tabular
                                          idx_start: int = 0,
                                          num_train: int = 1000, 
                                          num_val: int = 100, 
                                          num_test: int = 100, 
                                          all_value_names: list[str] = ["Age"],
                                          target_value_name: str = "Age",
                                          image_size: list[int] = [128, 128],
                                          replace_processed: bool = False,
                                          data_filtering: bool = True,):
    assert processed_table_pickle_dir is not None, "The name of the processed pickle file needs to be provided"
    assert biomarker_table_pickle_dir is not None, "The name of the final target pickle file needs to be provided"
    
    # Load the table with all selected subjects with all of the phenotypes
    if os.path.exists(processed_table_pickle_dir) and not replace_processed:
        print("Loading the processed table")
        with open(processed_table_pickle_dir, 'rb') as file:
            processed_table = pickle.load(file)
    else:
        # Find the subjects with enough sax slice number and proper image size
        if os.path.exists(subj_ids_dir):
            with open(subj_ids_dir, 'rb') as file:
                processed_ids_ = pickle.load(file)
        else:
            subjects_ids = find_indices_of_images(load_dir, sax_bbox_size=image_size, lax_bbox_size=image_size)
            processed_ids_ = process_cmr_images(load_dir=load_dir, 
                                                prep_dir=processed_dir, 
                                                file_name=processed_file_name,
                                                sax_bbox_size=image_size,
                                                lax_bbox_size=image_size, 
                                                replace_processed=replace_processed,
                                                id_list=subjects_ids) 
            processed_ids_ = sorted(processed_ids_) # Subjects with CMR images
        
        biomarker_table = pd.read_csv(biomarker_tabular_dir) # Load biomarker tabular data
        col_names = biomarker_table.columns
        all_value_names = list(set(all_value_names).intersection(set(col_names))) # Keep the onces in table
        processed_idx = biomarker_table["eid_87802"].isin(processed_ids_) # Cases in the image list
        processed_table_ = biomarker_table.loc[processed_idx, ["eid_87802"] + all_value_names]
        processed_table = processed_table_.dropna() # Remove nan
        if data_filtering: # Filter data based on the given table conditions
            processed_table = apply_table_conditions(processed_table, **table_condition_dict, 
                                                     condition_table_dir=all_feature_tabular_dir)
        processed_table.to_pickle(processed_table_pickle_dir)

    # Select the required target values
    column_keys = ["eid_87802"] + [target_value_name]
    target_table = processed_table[column_keys]
    target_table.to_pickle(biomarker_table_pickle_dir) # Save the selected indices with biomarkers
    subject_ids = target_table["eid_87802"]

    #####################################################        
    # Split training, validation, and test subject paths   
    subject_paths = []
    for parent in subject_ids:
        subject_path = Path(processed_dir) / str(parent) / Path(processed_file_name)
        subject_paths.append(subject_path)
            
    split = (num_train, num_val, num_test)
    topk_paths = subject_paths[idx_start : (sum(split) + idx_start)]
    train_idxs, val_idxs, test_idxs = [list(s) for s in random_split(list(range(len(topk_paths))), split)]
    train_paths = [topk_paths[i] for i in train_idxs]
    val_paths = [topk_paths[i] for i in val_idxs]
    test_paths = [topk_paths[i] for i in test_idxs]
    paths = {"train": train_paths, "val": val_paths, "test": test_paths}
    with open(path_dir, 'wb') as handle:
        pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return paths, target_table


def apply_table_conditions(table: pd.DataFrame, 
                           health_flag: int = 0, # 0: only have the healthy cases; 1: only have the unhealthy cases
                           sorting_with_age: bool = True,
                           condition_table_dir: Optional[str] = None,):
    """
    Generate a table containing the subject indices with its target biomarker values.
    The subjects in the table satifies the following conditions:
        - considered healthy,
        - have all required target cardiac biomarkers.
    """
    # Load healthy subjects indices ftom all feature tabular data
    assert condition_table_dir is not None, "All feature table is not provided."
    print("Selecting healthy subjects takes a couple of minutes")
    healthy_idx_ = find_healthy_subjects(condition_table_dir)
    idx_ = table["eid_87802"]
    if health_flag:
        idx = list(set(idx_) & set(healthy_idx_))
    else: # Only load the unhealthy cases
        idx = list(set(idx_) - set(healthy_idx_))
    table = table.loc[table["eid_87802"].isin(idx)]

    # Sort the table based on 5 age groups evenly
    if sorting_with_age:
        age_groups = np.arange(table["Age"].min()-5, table["Age"].max()+5, 5)
        table["Age_group"] = pd.cut(table["Age"], bins=age_groups, labels=age_groups[:-1])
        for age_group in age_groups[:-1]:
            num = len(table.loc[table["Age_group"] == age_group])
            table.loc[table["Age_group"] == age_group, "Age_group_idx"] = range(num)
        table = table.sort_values(by=["Age_group_idx", "Age_group"])

    return table
