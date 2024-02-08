"""
Completely ignores the graph structure and just applies an MLP to the pooled vertex features
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList, Dropout, Sequential
from torch_geometric.nn import global_add_pool, global_mean_pool

from Models.utils import get_pooling_fct, get_mlp, get_activation

class MLP(torch.nn.Module):
    """
    MLP model that completely ignores the edges and edge features.
    """
    def __init__(self, num_layers, node_encoder, emb_dim, num_classes, num_tasks, dropout_rate, graph_pooling, activation):
        super(MLP, self).__init__()

        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.node_encoder = node_encoder
        self.activation =  get_activation(activation)
        self.pool = get_pooling_fct(graph_pooling)
        self.node_level_mlp = get_mlp(num_layers=num_layers, 
                                      in_dim = emb_dim, 
                                      out_dim = emb_dim, 
                                      hidden_dim = emb_dim // 2, 
                                      activation = self.activation, 
                                      dropout_rate = dropout_rate)
        self.graph_level_mlp = get_mlp(num_layers = num_layers, 
                                       in_dim = emb_dim, 
                                       out_dim = num_tasks*num_classes, 
                                       hidden_dim = emb_dim // 2, 
                                       activation = self.activation, 
                                       dropout_rate = dropout_rate)
        
    def forward(self, data):
        x = self.node_encoder(data.x)
        x = self.node_level_mlp(x)
        x = self.pool(x, data.batch).float()
        x = self.graph_level_mlp(x)

        if self.num_tasks == 1:
            x = x.view(-1, self.num_classes)
        else:
            x.view(-1, self.num_tasks, self.num_classes)
        return x

    def __repr__(self):
        return self.__class__.__name__