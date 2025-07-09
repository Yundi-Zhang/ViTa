import os
from typing import List, Tuple
import random
import copy
import numpy as np
import torch

from datasets.datasets import Cardiac3DplusTAllAX
from utils.general import image_normalization


__all__ = ["ImagingAllAXTabular", "ImagingAllAXTabular_Test",
           "ImagingAllAXTabular_TabRcon", "ImagingAllAXTabular_TabRcon_Test",
           ]


class ImagingAllAXTabular(Cardiac3DplusTAllAX):
    def __init__(self, image_paths, tabular_data, marginal_distributions, labels_data, **kwargs):
        super().__init__(image_paths, **kwargs)

        # Image
        self.subject_paths = image_paths
        self.frame_to_keep = kwargs.get("frame_to_keep")

        # Tabular
        self.data_tabular = tabular_data
        self.marginal_distributions = marginal_distributions
        self.tab_augs = kwargs.get("tab_augment")
        self.tab_augmentation = self._tab_augment
        self.c = kwargs.get("corruption_rate")
        
        # Classifier
        self.labels = labels_data
    
    @property
    def _tab_augment(self) -> bool:
        return self.tab_augs

    def corrupt(self, subject):
        """
        Creates a copy of a subject, selects the indices 
        to be corrupted (determined by hyperparam corruption_rate)
        and replaces their values with ones sampled from marginal distribution
        """
        subject = copy.deepcopy(subject)

        indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
        for i in indices:
            sample = torch.randperm(self.marginal_distributions.shape[1])[0] 
            subject[i] = self.marginal_distributions[i, sample]
        return subject

    def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
        """
        One-hot encodes a subject's features
        """
        out = []
        for i in range(len(subject)):
            if self.field_lengths_tabular[i] == 1:
                out.append(subject[i].unsqueeze(0))
            else:
                out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
        return torch.cat(out)
    
    def load_sparse_im_arr(self, index):
        npy_path = self.subject_paths[index]
        assert os.path.exists(npy_path), f"File not found: {npy_path}"
        if npy_path.name[-4:] == ".npy":
            im_data = np.load(npy_path, allow_pickle=True).item()
        return im_data
    
    def load_img(self, index: int):
        # Load short axis and long axis images
        subject_id = self.get_subject_id(index)
        sax_slice_num = self.slice_num-3
        sax_im_data, _, lax_im_data, _ = self.load_im_seg_arr(index, z_num=sax_slice_num)
        assert lax_im_data.shape[:2] == sax_im_data.shape[:2], f"Subject id: {subject_id}"
        im_data = np.concatenate([lax_im_data, sax_im_data], axis=2)
        im_data = np.transpose(im_data, (2, 3, 0, 1)) # Move the slice and time dimension to the front
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)

        # Sample limited frames
        if self.frame_to_keep != im_data.shape[1]:
            step = (self.time_frame - 1) / (self.frame_to_keep - 1)
            indices = [round(i * step) for i in range(self.frame_to_keep)]
            im_data = im_data[:, indices, ...]
            
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        return im_data
    
    def load_img_sparse(self, index: int):

        im_data = self.load_sparse_im_arr(index) # TODO: temporary for sparse dataset 
        assert len(im_data.shape) == 4 and im_data.shape[0] == self.slice_num
        im_data = image_normalization(im_data)
        im_data = torch.from_numpy(im_data)
            
        if self.augmentation:
            im_data = self.apply_augmentations(im_data)
        return im_data

    def load_tab(self, index: int):
        # Load tabular data and labels
        tabular_data = self.data_tabular[index]
        if self.tab_augmentation:
            tabular_data = self.corrupt(tabular_data)
        return tabular_data
            
    def load_label(self, index: int):
        if isinstance(self.labels, torch.Tensor):
            label = self.labels[index].clone().detach()
        else:
            label = torch.tensor(self.labels[index], dtype=torch.long)
        return label
    
    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        return self.load_img(index), self.load_tab(index), self.load_label(index), self.get_subject_id(index)

    def __len__(self) -> int:
        return len(self.subject_paths)
    
    def get_view(self) -> int:
        return 2 # For both long and short-axis views
    
    
class ImagingAllAXTabular_Test(ImagingAllAXTabular):
    
    @property
    def _augment(self) -> bool:
        return False
    
    @property
    def _tab_augment(self) -> bool:
        return False
    
    
class ImagingAllAXTabular_TabRcon(ImagingAllAXTabular):
    def __init__(self, raw_tabular_data, **kwargs):
        super().__init__(**kwargs)
        self.raw_tabular = raw_tabular_data
    
    def load_raw_tab(self, index: int):
        # Load raw tabular data and labels
        raw_tabular_data = self.raw_tabular[index]
        return raw_tabular_data

    def __getitem__(self, index: int):
        return self.load_img(index), self.load_raw_tab(index), self.get_subject_id(index)


class ImagingAllAXTabular_TabRcon_Test(ImagingAllAXTabular_TabRcon):
    
    @property
    def _augment(self) -> bool:
        return False
    
    @property
    def _tab_augment(self) -> bool:
        return False
