"""
For the QM9 dataset, we are given a list of 19 separate targets. 
This transform allows us to remove all except the target from the dataset.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class SelectOnlyOneTarget(BaseTransform):
    r""" 
    Select only one target from data.y
    """
    def __init__(self, target):
        self.target = target

    def __call__(self, data: Data):
        data.y = torch.unsqueeze(data.y[:, self.target], 1)
        data.num_classes = 1
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.target})'