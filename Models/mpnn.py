import torch
from torch_geometric.nn import MessagePassing, GINEConv, GATv2Conv
from torch_geometric.utils import degree
from torch.nn import Linear, ReLU, ModuleList, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F

from Models.utils import get_pooling_fct, get_activation, get_mlp
from Misc.utils import PredictionType

def get_mp_layer(emb_dim, activation, mp_type):
    if mp_type.lower() == "gin":
        nn = Sequential(Linear(emb_dim, 2*emb_dim), BatchNorm1d(2*emb_dim), activation, Linear(2*emb_dim, emb_dim))
        return GINEConv(nn = nn)
    elif mp_type.lower() == "gcn":
        return GCNConv(emb_dim=emb_dim, activation=activation)
    elif mp_type.lower() == "gat":
        mp_layer = GATv2Conv(in_channels=emb_dim, out_channels=emb_dim, edge_dim=emb_dim, heads=3, concat=False)
        nn = Sequential(Linear(emb_dim, 2*emb_dim), BatchNorm1d(2*emb_dim), activation, Linear(2*emb_dim, emb_dim))
        return ConvWrapper(conv=mp_layer, nn=nn)
    raise Exception("Unknown message passing type")    
    
class MPNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer, emb_dim, 
                    gnn_type, residual, drop_ratio , JK, graph_pooling,
                    node_encoder, edge_encoder, num_mlp_layers, activation, prediction_type):
        """
        Message passing graph neural network.
        """
        super(MPNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.residual = residual
        self.JK = JK
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        self.activation = get_activation(activation)
        self.prediction_type = prediction_type
        
        assert self.num_layer >= 1
        
        # Todo: virtual node
        
        # Message Passing Layers
        print(f"Message Passing Layers: {gnn_type}")
        self.mp_layers, self.batch_norms = ModuleList([]), ModuleList([])
        self.dropout = Dropout(p=drop_ratio)
        for _ in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.mp_layers.append(get_mp_layer(emb_dim, self.activation, gnn_type))
            
        if prediction_type in [PredictionType.NODE_PREDICTION, PredictionType.GRAPH_PREDICTION]:
            print(f"Graph pooling function: {graph_pooling}")
            self.pool = get_pooling_fct(graph_pooling)
            self.mlp = get_mlp(num_layers=num_mlp_layers, 
                            in_dim=self.emb_dim, 
                            out_dim=self.num_classes*self.num_tasks, 
                            hidden_dim=self.emb_dim // 2, 
                            activation=self.activation, 
                            dropout_rate=drop_ratio)

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        edge_attr = self.edge_encoder(edge_attr)
        
        # Each entry is the embedding of all nodes per message passing layers 
        h_list = [self.node_encoder(x)]
        for layer, mp_layer in enumerate(self.mp_layers):
            h = mp_layer(x=h_list[layer], edge_index=edge_index, edge_attr=edge_attr)  
            h = self.batch_norms[layer](h)
            h = self.dropout(h)

            # No ReLU for last layer
            if layer != self.num_layer - 1:
                h = self.activation(h)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)
        
        # Todo: jumping knowledge
        h_node = h_list[-1]
        
       
        if self.prediction_type == PredictionType.NODE_EMBEDDING:
            return h_node
        
        elif self.prediction_type == PredictionType.NODE_PREDICTION:
            prediction = self.mlp(h_node)
            
        elif self.prediction_type == PredictionType.GRAPH_PREDICTION:
            h_graph = self.pool(h_node, batched_data.batch)
            prediction = self.mlp(h_graph)
        
        else: # PredictionType.EDGE_PREDICTION
            h_edge_endpoints = h_node[batched_data.edge_label_index]
            prediction = torch.sum(h_edge_endpoints[0,:,:] * h_edge_endpoints[1,:,:], dim=-1)
            
        # Reshape prediction to fit task
        if self.num_tasks == 1:
            prediction = prediction.view(-1, self.num_classes)
        else:
            prediction.view(-1, self.num_tasks, self.num_classes)
        return prediction

class ConvWrapper(torch.nn.Module):
    """
    Wrapper to combine a convolutional message passing layers with few neurons (e.g. GAT) together with larger MLPs
    """
    def __init__(self, conv, nn):
        super(ConvWrapper, self).__init__()
        self.conv = conv
        self.nn = nn
        
    def forward(self, x, edge_index, edge_attr):
        return self.nn(self.conv(x, edge_index, edge_attr))

class GCNConv(MessagePassing):
    """
    Adapted from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py (MIT License)
    """
    def __init__(self, emb_dim, activation):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.activation = activation

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_attr, norm=norm) + self.activation(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * self.activation(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out