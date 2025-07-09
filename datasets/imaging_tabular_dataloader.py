import math
from pathlib import Path

import random
import pickle
from typing import List
import lightning.pytorch as pl
import pandas as pd
from termcolor import colored
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Sampler, SubsetRandomSampler

from datasets.imaging_preprocessing import load_image_paths
from datasets.imaging_tabular_datasets import *
    

def get_stratified_sampler(dataset, num_samples):
    """
    Returns a SubsetRandomSampler that samples `num_samples` from the dataset
    while preserving the class distribution (i.e., stratified sampling).
    
    Parameters:
        dataset: a dataset object with .raw_tabular containing class labels
        num_samples: total number of samples to draw
    """
    # Get targets from the dataset
    targets = np.array(dataset.raw_tabular)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=num_samples)
    train_idx, _ = next(sss.split(np.zeros(len(targets)), targets))
    
    return SubsetRandomSampler(train_idx)


def get_balanced_sampler(dataset):
    """
    Returns a SubsetRandomSampler that samples an equal number of examples from each class.
    
    Parameters:
        dataset: a dataset object with dataset.raw_tabular containing class labels.
    """
    targets = dataset.raw_tabular

    # Get indices where label == 1.0 (positives) and label == 0.0 (negatives)
    pos_indices = torch.where(targets == 1.0)[0].tolist()
    neg_indices = torch.where(targets == 0.0)[0].tolist()
    
    num_pos = len(pos_indices)

    if len(neg_indices) < num_pos:
        raise ValueError(f"Not enough negative samples to match {num_pos} positives.")
    
    # Randomly sample negatives
    sampled_neg_indices = random.sample(neg_indices, num_pos)

    # Combine and shuffle
    balanced_indices = pos_indices + sampled_neg_indices
    random.shuffle(balanced_indices)

    return SubsetRandomSampler(balanced_indices)

    
class RandomDistributedSampler(Sampler):
    def __init__(self, dataset, num_samples=None, replacement=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement

        # Check if distributed is available and initialized
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("distributed is not available and initialized")
            self.num_replicas = 1
            self.rank = 0

        self.num_samples_per_rank = math.ceil(self.num_samples / self.num_replicas)
        self.total_size = self.num_samples_per_rank * self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)  # Ensuring reproducibility

        # Randomly sample indices from the entire dataset
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # Ensure each rank gets a non-overlapping subset of indices
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # Truncate to desired number of samples per rank
        return iter(indices[:self.num_samples_per_rank])

    def __len__(self):
        return self.num_samples_per_rank


class CMRImageTabularDataModule(pl.LightningDataModule):
    def __init__(self, load_dir: str, # Image
                 processed_dir: str,  # Image
                 dataloader_image_file_folder: str, # Image
                 dataloader_tabular_file_folder: str, # Image
                 cmr_path_pickle_name: str, # Image
                 subj_ids_with_required_size_pickle_name: str, # Image
                 ):
        super().__init__()
        
        # For image loading
        self.load_dir = load_dir
        self.processed_dir = processed_dir
        self.dataloader_image_file_folder = dataloader_image_file_folder
        self.dataloader_tabular_file_folder = dataloader_tabular_file_folder
        self.cmr_path_pickle_dir = Path(dataloader_image_file_folder) / cmr_path_pickle_name
        self.subj_ids_dir = Path(dataloader_image_file_folder) / subj_ids_with_required_size_pickle_name
        
        # For validation labels loading
        self.validation_form = None
        self.selected_cols = None
        
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        Path(dataloader_image_file_folder).mkdir(parents=True, exist_ok=True)
        Path(dataloader_tabular_file_folder).mkdir(parents=True, exist_ok=True)
        
    def setup(self, stage, multi_gpu: bool = False, **kwargs):
        if self.train_dset is not None and self.val_dset is not None:
            # trainer.fit seems to call datamodule.setup(), we don't wanna do it twice because our
            # model was already given the val_dset and we don't want it to change  
            return    
        # -----------------------------------------------------------------------------------------------
        # Load image paths
        paths = load_image_paths(path_dir=self.cmr_path_pickle_dir, subj_ids_dir=self.subj_ids_dir,
                                 load_dir=self.load_dir, processed_dir=self.processed_dir, 
                                 **kwargs)
        num_train = kwargs.get("num_train")
        num_val = kwargs.get("num_val")
        num_test = kwargs.get("num_test")
        
        n_train = num_train if num_train < len(paths["train"]) else len(paths["train"])
        n_val = num_val if num_val < len(paths["val"]) else len(paths["val"])
        n_test = num_test if num_test < len(paths["test"]) else len(paths["test"])
        
        train_image_paths=paths["train"][:n_train]
        val_image_paths=paths["val"][:n_val]
        test_image_paths=paths["test"][:n_test]
        
        with open("train_paths.pkl", 'wb') as file_new:
            pickle.dump(train_image_paths, file_new)
        with open("val_paths.pkl", 'wb') as file_new:
            pickle.dump(val_image_paths, file_new)
        with open("test_paths.pkl", 'wb') as file_new:
            pickle.dump(test_image_paths, file_new)

        # -----------------------------------------------------------------------------------------------
        # Load tabular data and pairing tabular and imaging data in the same order
        tabular_path = Path(self.dataloader_tabular_file_folder) / kwargs.get("tabular_data")
        raw_tabular_path = Path(self.dataloader_tabular_file_folder) / kwargs.get("raw_tabular_data")
        train_sorted_tabular_data = self.load_filter_tabular_data(train_image_paths, tabular_path)
        val_sorted_tabular_data = self.load_filter_tabular_data(val_image_paths, tabular_path)
        test_sorted_tabular_data = self.load_filter_tabular_data(test_image_paths, tabular_path)
        
        train_margin = torch.permute(train_sorted_tabular_data, (1, 0))
        val_margin = torch.permute(val_sorted_tabular_data, (1, 0))
        test_margin = torch.permute(test_sorted_tabular_data, (1, 0))
        
        train_sorted_raw_tabular_data = self.load_filter_tabular_data(train_image_paths, raw_tabular_path)
        val_sorted_raw_tabular_data = self.load_filter_tabular_data(val_image_paths, raw_tabular_path)
        test_sorted_raw_tabular_data = self.load_filter_tabular_data(test_image_paths, raw_tabular_path)

        # -----------------------------------------------------------------------------------------------
        # Load classifier
        self.validation_form = kwargs.get("validation_form")
        self.selected_cols = kwargs.get("selected_cols")
        if self.validation_form == "classification":
            labels_path = Path(self.dataloader_tabular_file_folder) / kwargs.get("labels")
            train_labels_data = self.load_filter_tabular_data(train_image_paths, labels_path)
            val_labels_data = self.load_filter_tabular_data(val_image_paths, labels_path)
            test_labels_data = self.load_filter_tabular_data(test_image_paths, labels_path)
        elif self.validation_form == "visualization":
            selected_cols = kwargs.get("selected_cols")
            raw_labels_path = Path(self.dataloader_tabular_file_folder) / kwargs.get("labels")
            train_labels_data = self.load_filter_tabular_data(train_image_paths, raw_labels_path, selected_cols)
            val_labels_data = self.load_filter_tabular_data(val_image_paths, raw_labels_path, selected_cols)
            test_labels_data = self.load_filter_tabular_data(test_image_paths, raw_labels_path, selected_cols)
        else:
            raise NotImplementedError(f"The validation form {self.validation_form} is not implemented")
        # -----------------------------------------------------------------------------------------------
        dataset_cls = kwargs.get("dataset_cls")
        self.train_dset = eval(f'{dataset_cls}')(image_paths=train_image_paths, 
                                                 ignore_tab=kwargs.get("ignore_phenotype_tabular"), 
                                                 load_seg=kwargs.get("load_seg"), 
                                                 augs=kwargs.get("augment"), 
                                                 sax_slice_num=kwargs.get("sax_slice_num"),
                                                 t_downsampling_ratio=kwargs.get("t_downsampling_ratio"),
                                                 frame_to_keep=kwargs.get("frame_to_keep"),
                                                 # tabular
                                                 tabular_data=train_sorted_tabular_data,
                                                 raw_tabular_data=train_sorted_raw_tabular_data,
                                                 corruption_rate=kwargs.get("corruption_rate"),
                                                 tab_augment=kwargs.get("tab_augment"),
                                                 marginal_distributions=train_margin,
                                                 one_hot_tabular=kwargs.get("one_hot_tabular"),
                                                 # label
                                                 labels_data=train_labels_data,
                                                 ) 
        self.val_dset = eval(f'{dataset_cls}_Test')(image_paths=val_image_paths, 
                                                    ignore_tab=kwargs.get("ignore_phenotype_tabular"), 
                                                    load_seg=kwargs.get("load_seg"), 
                                                    augs=kwargs.get("augment"), 
                                                    sax_slice_num=kwargs.get("sax_slice_num"),
                                                    t_downsampling_ratio=kwargs.get("t_downsampling_ratio"), 
                                                    frame_to_keep=kwargs.get("frame_to_keep"),
                                                    # tabular
                                                    tabular_data=val_sorted_tabular_data,
                                                    raw_tabular_data=val_sorted_raw_tabular_data,
                                                    corruption_rate=kwargs.get("corruption_rate"),
                                                    marginal_distributions=val_margin,
                                                    one_hot_tabular=kwargs.get("one_hot_tabular"),
                                                    # label
                                                    labels_data=val_labels_data,
                                                    ) 
        self.test_dset = eval(f'{dataset_cls}_Test')(image_paths=test_image_paths,
                                                     ignore_tab=kwargs.get("ignore_phenotype_tabular"), 
                                                     load_seg=kwargs.get("load_seg"), 
                                                     augs=kwargs.get("augment"), 
                                                     sax_slice_num=kwargs.get("sax_slice_num"),
                                                     t_downsampling_ratio=kwargs.get("t_downsampling_ratio"), 
                                                     frame_to_keep=kwargs.get("frame_to_keep"),
                                                     # tabular
                                                     tabular_data=test_sorted_tabular_data,
                                                     raw_tabular_data=test_sorted_raw_tabular_data,
                                                     corruption_rate=kwargs.get("corruption_rate"),
                                                     marginal_distributions=test_margin,
                                                     one_hot_tabular=kwargs.get("one_hot_tabular"),
                                                     # label
                                                     labels_data=test_labels_data,
                                                     ) 
        if kwargs.get("train_num_per_epoch") is not None:
            if multi_gpu:
                trainer_sampler = RandomDistributedSampler(self.train_dset, num_samples=kwargs.get("train_num_per_epoch"))
                print(colored(f"num gpus:", 'blue', None, ['bold']), torch.cuda.device_count())
                print(colored(f"CUDA  available:", 'blue', None, ['bold']), torch.cuda.is_available())
            else:
                if kwargs.get("stratified_sampler"):
                    trainer_sampler = get_stratified_sampler(self.train_dset, num_samples=kwargs.get("train_num_per_epoch"))
                else:
                    trainer_sampler = RandomSampler(self.train_dset, num_samples=kwargs.get("train_num_per_epoch"))
        elif kwargs.get("balanced_sampler"):
            trainer_sampler = get_balanced_sampler(self.train_dset)
        else:
            trainer_sampler = None
        self._train_dataloader = DataLoader(self.train_dset, 
                                            batch_size=kwargs.get("batch_size"),
                                            sampler=trainer_sampler,
                                            num_workers=kwargs.get("num_workers"), 
                                            pin_memory=True,
                                            persistent_workers=kwargs.get("num_workers") > 0)
        self._val_dataloader = DataLoader(self.val_dset, batch_size=kwargs.get("batch_size"), num_workers=0)
        self._test_dataloader = DataLoader(self.test_dset, batch_size=kwargs.get("batch_size"), num_workers=0)

        print(colored(f"Preparing the training dataset: ", 'green', None, ['bold']), f"{dataset_cls}")
        print(colored(f"Training subjects number:", 'green', None, ['bold']), f"{len(train_image_paths)}")
        print(colored(f"Preparing the validation dataset: ", 'green', None, ['bold']), f"{dataset_cls}_Test")
        print(colored(f"Validation subjects number:", 'green', None, ['bold']), f"{len(val_image_paths)}")
        print(colored(f"Preparing the test dataset: ", 'green', None, ['bold']), f"{dataset_cls}_Test")
        print(colored(f"Testing subjects number:", 'green', None, ['bold']), f"{len(test_image_paths)}") # TODO
        
    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def load_filter_tabular_data(self, image_paths, tabular_path, selected_col_names: List[str] = None):
        """ Sort the tabular data based on subject ids so that the order of data is the same as for the subject order
        in imaging paths """
        tab_csv = pd.read_csv(tabular_path, index_col=None)
        imaging_id_list = [int(path.parent.name) for path in image_paths]
        filtered_tab = tab_csv[tab_csv["eid"].isin(imaging_id_list)]
        ordered_tab = filtered_tab.set_index("eid").reindex(imaging_id_list).reset_index()
        if selected_col_names != None:
            ordered_tab = ordered_tab[selected_col_names]
            bool_columns = ordered_tab.columns[ordered_tab.dtypes == bool]
            ordered_tab[bool_columns] = ordered_tab[bool_columns].astype(int)
            ordered_tab_values = ordered_tab.values
        else:
            ordered_tab_values = ordered_tab.drop("eid", axis=1).values
        ordered_tab_data = torch.tensor(ordered_tab_values, dtype=torch.float32)
        return ordered_tab_data
    
    def remove_a_column(self, index, input):
        mask = torch.ones(input.shape[1], dtype=torch.bool)
        mask[index] = False
        output = input[:, mask]
        return output
    