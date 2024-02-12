import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch.nn import Identity, Dropout, BatchNorm1d, Sequential, Linear

def get_activation(activation):
    match activation:
        case 'relu':
            return torch.nn.ReLU()
        case 'elu':
            return torch.nn.ELU()
        case 'id':
            return torch.nn.Identity()
        case 'sigmoid':
            return torch.nn.Sigmoid()
        case 'tanh':
            return torch.nn.Tanh()
        case 'gelu':
            return torch.nn.GELU()
        case _:
            raise NotImplementedError(f"Activation {activation} not implemented")

def get_pooling_fct(readout):    
    if readout == "sum":
        return global_add_pool
    elif readout == "mean":
        return global_mean_pool
    elif readout == "max":
        return global_max_pool
    else:
        raise NotImplementedError(f"Readout {readout} is not currently supported.")

def get_mlp(num_layers, in_dim, out_dim, hidden_dim, activation, dropout_rate):
    layers = []
    for i in range(num_layers):
        in_size = hidden_dim if i > 0 else in_dim
        out_size = hidden_dim if i < num_layers - 1 else out_dim

        layers.append(Linear(in_size, out_size))
        layers.append(BatchNorm1d(out_size))
                    
        if num_layers > 0 and i < num_layers - 1:
            layers.append(Dropout(p=dropout_rate))
            layers.append(activation)
            
    return Sequential(*layers)