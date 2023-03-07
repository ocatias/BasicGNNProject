import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, OneHotDegree

class DropFeatures(BaseTransform):
    r""" 
    Drop vertex and edge features from graph
    """
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.one_hot_encoder = OneHotDegree(emb_dim - 1)

    def __call__(self, data: Data):
        data.x = torch.zeros([data.x.shape[0], 0], dtype=torch.float32)
        data = self.one_hot_encoder(data)
        data.edge_attr = torch.zeros([data.edge_attr.shape[0], self.emb_dim], dtype=torch.float32)
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.emb_dim})'