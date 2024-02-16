
import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

class NodeEncoder(torch.nn.Module):
    """
    Adapted from https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py (MIT License)
    """

    def __init__(self, emb_dim, feature_dims = None):
        super(NodeEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()
        if feature_dims is None:
            feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        x = x.long()
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class EdgeEncoder(torch.nn.Module):
    """
    Adapted from https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py (MIT License)
    """
    
    def __init__(self, emb_dim, feature_dims = None):
        super(EdgeEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        if feature_dims is None:
            feature_dims = get_bond_feature_dims()

        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        edge_attr = edge_attr.long()
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   
     
"""
From github.com/rampasek/GraphGPS (MIT License)
"""
       
VOC_node_input_dim = 14
# VOC_edge_input_dim = 1 or 2; defined in class VOCEdgeEncoder

class VOCNodeEncoder(torch.nn.Module):
    """
    Encoder for the PASCALVOC-SP dataset
    From github.com/rampasek/GraphGPS (MIT License)
    """
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, x):
        return self.encoder(x)


class VOCEdgeEncoder(torch.nn.Module):
    """
    Encoder for the PASCALVOC-SP dataset
    From github.com/rampasek/GraphGPS (MIT License)
    """
    def __init__(self, emb_dim, edge_wt_region_boundary=True):
        super().__init__()

        VOC_edge_input_dim = 2 if edge_wt_region_boundary else 1
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, edge_attr):
        return self.encoder(edge_attr)