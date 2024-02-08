import torch
from torch_geometric.nn import MessagePassing, GINEConv, GATv2Conv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree
from torch.nn import Linear, ReLU, ModuleList, Sequential, BatchNorm1d, Dropout
import torch.nn.functional as F

class MPNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer, emb_dim, 
                    gnn_type, residual, drop_ratio , JK, graph_pooling,
                    node_encoder, edge_encoder, num_mlp_layers):
        '''
            
        '''

        super(MPNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.residual = residual
        self.JK = JK
        self.node_encoder = node_encoder
        self.edge_encoder = edge_encoder
        
        assert self.num_layer >= 1
        
        # Todo: virtual node
        
        
        # Message Passing Layers
        print(f"Message Passing Layers: {gnn_type}")
        self.mp_layers, self.batch_norms = ModuleList([]), ModuleList([])
        self.dropout = Dropout(p=drop_ratio)
        for _ in range(self.num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            # Todo: GCN          
            if gnn_type.lower() == "gin":
                nn = Sequential(Linear(emb_dim, 2*emb_dim), BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
                self.mp_layers.append(GINEConv(nn = nn))
            elif gnn_type.lower() == "gcn":
                self.mp_layers.append(GCNConv(emb_dim=emb_dim))
            elif gnn_type.lower() == "gat":
                self.mp_layers.append(GATv2Conv(in_channels=emb_dim, out_channels=emb_dim, edge_dim=emb_dim))
            else:
                raise NotImplementedError

        # Graph Pooling
        print(f"Pooling operation: {graph_pooling}")
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise NotImplementedError
        
        # MLP
        mlp = []
        hidden_size = self.emb_dim // 2
        for i in range(num_mlp_layers):
            in_size = hidden_size if i > 0 else self.emb_dim
            out_size = hidden_size if i < num_mlp_layers - 1 else self.num_classes*self.num_tasks

            mlp.append(Linear(in_size, out_size))
            mlp.append(BatchNorm1d(out_size))
                        
            if num_mlp_layers > 0 and i < num_mlp_layers - 1:
                mlp.append(self.dropout)
                mlp.append(ReLU())
                
        self.mlp = Sequential(*mlp)

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
                h = F.relu(h)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)
        
        # Todo: jumping knowledge
        h_node = h_list[-1]
        h_graph = self.pool(h_node, batched_data.batch)

        # Final MLP for predictions
        prediction = self.mlp(h_graph)

        # Reshape prediction to fit task
        if self.num_tasks == 1:
            prediction = prediction.view(-1, self.num_classes)
        else:
            prediction.view(-1, self.num_tasks, self.num_classes)
        return prediction

class GCNConv(MessagePassing):
    """
    Adapted from https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py (MIT License)
    """
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_attr, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

if __name__ == '__main__':
    MPNN(num_tasks = 10)