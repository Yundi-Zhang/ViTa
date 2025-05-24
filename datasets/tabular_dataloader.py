import csv
import math
from pathlib import Path

import pickle
from typing import List, Optional
import lightning.pytorch as pl
import pandas as pd
from termcolor import colored
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Sampler

from datasets.imaging_tabular_dataloader import RandomDistributedSampler
from datasets.tabular_datasets import *


class TabularDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataloader_image_file_folder: str,
                 dataloader_tabular_file_folder: str, # Image
                 cmr_path_pickle_name: str, # Image
                #  tabular_root_folder: str, # Tabular
                #  labels_root_folder: str, # Labels
                 ):
        super().__init__()
        
        self.dataloader_tabular_file_folder = dataloader_tabular_file_folder
        self.cmr_path_pickle_dir = Path(dataloader_image_file_folder) / cmr_path_pickle_name
        
        self.validation_form = None
        self.selected_cols = None
        self.train_dset = None
        self.val_dset = None
        self.test_dset = None
        Path(dataloader_image_file_folder).mkdir(parents=True, exist_ok=True)
        
    def setup(self, stage, multi_gpu: bool = False, **kwargs):
        if self.train_dset is not None and self.val_dset is not None:
            # trainer.fit seems to call datamodule.setup(), we don't wanna do it twice because our
            # model was already given the val_dset and we don't want it to change  
            return    
             
        # Load image paths for id alignment
        with open(self.cmr_path_pickle_dir, "rb") as file:
            paths = pickle.load(file)
            
        num_train = kwargs.get("num_train")
        num_val = kwargs.get("num_val")
        num_test = kwargs.get("num_test")
        
        n_train = num_train if num_train < len(paths["train"]) else len(paths["train"])
        n_val = num_val if num_val < len(paths["val"]) else len(paths["val"])
        n_test = num_test if num_test < len(paths["test"]) else len(paths["test"])
        
        train_image_paths=paths["train"][:n_train]
        val_image_paths=paths["val"][:n_val]
        test_image_paths=paths["test"][:n_test]

        # -----------------------------------------------------------------------------------------------
        # Load tabular data and pairing tabular and imaging data in the same order
        tabular_path = Path(self.dataloader_tabular_file_folder) / kwargs.get("tabular_data")
        train_sorted_tabular_data, train_margin = self.load_filter_tabular_data(train_image_paths, tabular_path)
        val_sorted_tabular_data, val_margin = self.load_filter_tabular_data(val_image_paths, tabular_path)
        # test_sorted_tabular_data, test_margin = self.load_filter_tabular_data(test_image_paths, tabular_path) # TODO
        # -----------------------------------------------------------------------------------------------
        # Load classifier
        self.validation_form = kwargs.get("validation_form")
        self.selected_cols = kwargs.get("selected_cols")
        if self.validation_form == "classification":
            labels_path = Path(self.dataloader_tabular_file_folder) / kwargs.get("labels")
            train_labels_data, _ = self.load_filter_tabular_data(train_image_paths, labels_path)
            val_labels_data, _ = self.load_filter_tabular_data(val_image_paths, labels_path)
            # test_labels_data, _ = self.load_filter_tabular_data(test_image_paths, labels_path) # TODO
        elif self.validation_form == "visualization":
            selected_cols = kwargs.get("selected_cols")
            raw_labels_path = kwargs.get("labels")
            train_labels_data, _ = self.load_filter_tabular_data(train_image_paths, raw_labels_path, selected_cols)
            val_labels_data, _ = self.load_filter_tabular_data(val_image_paths, raw_labels_path, selected_cols)
            # test_labels_data, _ = self.load_filter_tabular_data(test_image_paths, raw_labels_path, selected_cols) # TODO
        else:
            raise NotImplementedError(f"The validation form {self.validation_form} is not implemented")
        # -----------------------------------------------------------------------------------------------
        # field_lengths_tabular_path = Path(self.labels_root_folder) / kwargs.get("field_lengths_tabular")
        dataset_cls = kwargs.get("dataset_cls")
        print(colored(f"Preparing the training dataset: ", 'green', None, ['bold']), f"{dataset_cls}")
        print(colored(f"Training subjects number:", 'green', None, ['bold']), f"{n_train}")
        self.train_dset = eval(f'{dataset_cls}')(tabular_data=train_sorted_tabular_data,
                                                 marginal_distributions=train_margin,
                                                 labels_data=train_labels_data,
                                                 corruption_rate=kwargs.get("corruption_rate"),
                                                 one_hot_tabular=kwargs.get("one_hot_tabular"),
                                                 augs=kwargs.get("tab_augment"), 
                                                 ) 
        print(colored(f"Preparing the validation dataset: ", 'green', None, ['bold']), f"{dataset_cls}_Test")
        print(colored(f"Validation subjects number:", 'green', None, ['bold']), f"{n_val}")
        self.val_dset = eval(f'{dataset_cls}_Test')(tabular_data=val_sorted_tabular_data,
                                                    marginal_distributions=val_margin,
                                                    labels_data=val_labels_data,
                                                    corruption_rate=kwargs.get("corruption_rate"),
                                                    one_hot_tabular=kwargs.get("one_hot_tabular"),
                                                    augs=kwargs.get("tab_augment"), 
                                                    ) 
        print(colored("The test dataset is replaced by validation as we don't have test tabular data yet", 'red', None, ['bold']))
        print(colored(f"Preparing the test dataset: ", 'green', None, ['bold']), f"{dataset_cls}_Test")
        print(colored(f"Testing subjects number:", 'green', None, ['bold']), f"{n_val}") # TODO
        self.test_dset = eval(f'{dataset_cls}_Test')(tabular_data=val_sorted_tabular_data, # TODO
                                                     marginal_distributions=val_margin, # TODO
                                                     labels_data=val_labels_data, # TODO
                                                     corruption_rate=kwargs.get("corruption_rate"),
                                                     one_hot_tabular=kwargs.get("one_hot_tabular"),
                                                     augs=kwargs.get("tab_augment"), 
                                                     ) 
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
            ordered_tab_values = ordered_tab.values
        else:
            ordered_tab_values = ordered_tab.drop("eid", axis=1).values
        ordered_tab_data = torch.tensor(ordered_tab_values, dtype=torch.float32)
        margin = torch.permute(ordered_tab_data, (1, 0))
        
        return ordered_tab_data, margin
