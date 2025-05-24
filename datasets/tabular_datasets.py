from typing import List, Tuple
import random
import copy

import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.io import read_image

from torch.utils.data import Dataset
from utils.general import image_normalization


__all__ = ["UKBBTabular", "UKBBTabular_Test"]


class UKBBTabular(Dataset):
    def __init__(self, tabular_data, marginal_distributions, labels_data, **kwargs):
        super().__init__()
        # Tabular
        self.data_tabular = tabular_data
        self.marginal_distributions = marginal_distributions

        self.c = kwargs.get("corruption_rate")
        self.one_hot_tabular = kwargs.get("one_hot_tabular")
        self.augs = kwargs.get("augs", True)
        self.augmentation = self._augment
        if self.one_hot_tabular:
            field_lengths_tabular_path = kwargs.get("field_lengths_tabular")
            self.field_lengths_tabular = torch.load(field_lengths_tabular_path)
        
        # Classifier
        self.labels = labels_data
        
    @property
    def _augment(self) -> bool:
        return self.augs
    
    def get_input_size(self) -> int:
        """
        Returns the number of fields in the table. 
        Used to set the input number of nodes in the MLP
        """
        if self.one_hot_tabular:
            return int(sum(self.field_lengths_tabular))
        else:
            return len(self.data[0])

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
    
    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        # Load tabular data and labels
        tabular_data = self.data_tabular[index]
        if self.augmentation:
            tabular_data = self.corrupt(tabular_data)
        if self.one_hot_tabular:
            tabular_data = self.one_hot_encode(tabular_data)

        if isinstance(self.labels, torch.Tensor):
            label = self.labels[index].clone().detach()
            label = label.to(dtype=torch.int64)
        else:
            label = torch.tensor(self.labels[index], dtype=torch.int64)
        
        return tabular_data, label, index

    def __len__(self) -> int:
        return self.data_tabular.shape[0]
    
    
class UKBBTabular_Test(UKBBTabular):
    
    @property
    def _augment(self) -> bool:
        return False
