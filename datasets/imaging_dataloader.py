from pathlib import Path

import pickle
from typing import List
import lightning.pytorch as pl
import pandas as pd
from termcolor import colored
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets.datasets import *
from datasets.datasets import AbstractDataset, AbstractDataset_Test
from datasets.imaging_tabular_dataloader import RandomDistributedSampler


class CMRImageDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataloader_image_file_folder: str,
                 dataloader_tabular_file_folder: str,
                 cmr_path_pickle_path: str,
                 ):
        super().__init__()
        
        self.dataloader_image_file_folder = dataloader_image_file_folder
        self.dataloader_tabular_file_folder = dataloader_tabular_file_folder
        self.cmr_path_pickle_path = cmr_path_pickle_path
        
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        Path(dataloader_image_file_folder).mkdir(parents=True, exist_ok=True)
        Path(dataloader_tabular_file_folder).mkdir(parents=True, exist_ok=True)
        
    def setup(self, stage, multi_gpu: bool = False, **kwargs):
        if self.train_dset is not None and self.val_dset is not None:
            # trainer.fit seems to call datamodule.setup(), we don't wanna do it twice because our
            # model was already given the val_dset and we don't want it to change  
            return
        # -----------------------------------------------------------------------------------------------
        # Load image paths
        with open(self.cmr_path_pickle_path, 'rb') as file:
            paths = pickle.load(file)
        imgs_n = len(paths["train"]) + len(paths["val"]) + len(paths["test"])
        print(f"Loaded {imgs_n} images from {self.cmr_path_pickle_path}.")
        num_train = kwargs.get("num_train")
        num_val = kwargs.get("num_val")
        num_test = kwargs.get("num_test")
        
        n_train = num_train if num_train < len(paths["train"]) else len(paths["train"])
        n_val = num_val if num_val < len(paths["val"]) else len(paths["val"])
        n_test = num_test if num_test < len(paths["test"]) else len(paths["test"])
        
        train_image_paths = paths["train"][:n_train]
        val_image_paths = paths["val"][:n_val]
        test_image_paths = paths["test"][:n_test]
        
        # -----------------------------------------------------------------------------------------------
        # Load tabular data and pairing tabular and imaging data in the same order
        if kwargs.get("ignore_phenotype_tabular"):
            train_sorted_tabular_data, val_sorted_tabular_data, test_sorted_tabular_data = None, None, None
        else:
            target_tabular_data = Path(self.dataloader_tabular_file_folder) / kwargs.get("target_tabular_data")
            train_sorted_tabular_data = self.load_filter_tabular_data(train_image_paths, target_tabular_data, selected_col_names=kwargs.get("target_value_name"))
            val_sorted_tabular_data = self.load_filter_tabular_data(val_image_paths, target_tabular_data, selected_col_names=kwargs.get("target_value_name"))
            test_sorted_tabular_data = self.load_filter_tabular_data(test_image_paths, target_tabular_data, selected_col_names=kwargs.get("target_value_name"))
        
        dataset_cls = kwargs.get("dataset_cls")
        self.train_dset = eval(f'{dataset_cls}')(train_image_paths, 
                                                 ignore_tab=kwargs.get("ignore_phenotype_tabular"),
                                                 target_table=train_sorted_tabular_data, 
                                                 target_value_name=kwargs.get("target_value_name"),
                                                 load_seg=kwargs.get("load_seg"), 
                                                 augs=kwargs.get("augment"), 
                                                 sax_slice_num=kwargs.get("sax_slice_num"),
                                                 t_downsampling_ratio=kwargs.get("t_downsampling_ratio"),
                                                 frame_to_keep=kwargs.get("frame_to_keep"))
        self.val_dset = eval(f'{dataset_cls}_Test')(val_image_paths, 
                                                    ignore_tab=kwargs.get("ignore_phenotype_tabular"),
                                                    target_table=val_sorted_tabular_data, 
                                                    target_value_name=kwargs.get("target_value_name"),
                                                    load_seg=kwargs.get("load_seg"), 
                                                    augs=kwargs.get("augment"), 
                                                    sax_slice_num=kwargs.get("sax_slice_num"),
                                                    t_downsampling_ratio=kwargs.get("t_downsampling_ratio"),
                                                    frame_to_keep=kwargs.get("frame_to_keep"))
        self.test_dset = eval(f'{dataset_cls}_Test')(test_image_paths, 
                                                     ignore_tab=kwargs.get("ignore_phenotype_tabular"),
                                                     target_table=test_sorted_tabular_data, 
                                                     target_value_name=kwargs.get("target_value_name"),
                                                     load_seg=kwargs.get("load_seg"), 
                                                     augs=kwargs.get("augment"), 
                                                     sax_slice_num=kwargs.get("sax_slice_num"),
                                                     t_downsampling_ratio=kwargs.get("t_downsampling_ratio"),
                                                     frame_to_keep=kwargs.get("frame_to_keep"))
        if kwargs.get("train_num_per_epoch") is not None:
            if multi_gpu:
                trainer_sampler = RandomDistributedSampler(self.train_dset, num_samples=kwargs.get("train_num_per_epoch"))
                print(colored(f"num gpus:", 'blue', None, ['bold']), torch.cuda.device_count())
                print(colored(f"CUDA  available:", 'blue', None, ['bold']), torch.cuda.is_available())
            else:
                trainer_sampler = RandomSampler(self.train_dset, num_samples=kwargs.get("train_num_per_epoch"))
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
        print(colored(f"Preparing the Testing dataset: ", 'green', None, ['bold']), f"{dataset_cls}_Test")
        print(colored(f"Testing subjects number:", 'green', None, ['bold']), f"{len(test_image_paths)}")

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
            if isinstance(selected_col_names, str):
                selected_col_names = [selected_col_names]
            col_names = ["eid"] + selected_col_names
            ordered_tab = ordered_tab[col_names]
            bool_columns = ordered_tab.columns[ordered_tab.dtypes == bool]
            ordered_tab[bool_columns] = ordered_tab[bool_columns].astype(int)
        else:
            ordered_tab = ordered_tab.drop("eid", axis=1)
        return ordered_tab
