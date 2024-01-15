import torch
from torch import cat
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import degree
from torch_geometric.nn import GATv2Conv

import math

from torch_scatter import scatter_mean

from Misc.transform_to_k_wl import k_wl_sequential_layers
from Misc.utils import num_connected_components
from Models.encoder import KWlEmbeddings, NodeEncoder
from Models.utils import avg_pool_custom, device


### GAT
# todo: implement GAT 
# class GINConv(MessagePassing):
#     def __init__(self, emb_dim, edge_encoder):
#         '''
#             emb_dim (int): node embedding dimensionality
#         '''

#         super(GINConv, self).__init__(aggr = "add")

#         self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
#         self.eps = torch.nn.Parameter(torch.Tensor([0]))
#         self.edge_encoder = edge_encoder

#     def forward(self, x, edge_index, edge_attr):
#         edge_embedding = self.edge_encoder(edge_attr)
#         out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

#         return out

#     def message(self, x_j, edge_attr):

#         # print(f"{x_j.shape}, {edge_attr.shape}")
#         return F.relu(x_j + edge_attr)

#     def update(self, aggr_out):
#         return aggr_out

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, edge_encoder, addition_input_dim=0):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim + addition_input_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = edge_encoder

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        if x_j.shape != edge_attr.shape:
            edge_attr = cat([edge_attr, edge_attr], dim=1)
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, edge_encoder):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = edge_encoder

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                 node_encoder=lambda x, _: x, edge_encoder=lambda x: x, k_wl=0, sequential_k_wl=False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.sequential_k_wl = sequential_k_wl
        self.k_wl = k_wl
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.gnn_type = gnn_type
        self.edge_encoder = edge_encoder
        for layer in range(num_layer):
            if self.sequential_k_wl and layer in k_wl_sequential_layers(num_layer, self.k_wl):
                addition_input_dim = emb_dim
            else:
                addition_input_dim = 0
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_encoder, addition_input_dim=addition_input_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_encoder))
            elif gnn_type == 'gat':
                self.convs.append(GATv2Conv(in_channels=emb_dim, out_channels=emb_dime, edge_dim=1))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        k_wl_layers = []
        if self.sequential_k_wl:
            seq_x = [batched_data[f'iso_type_{i}'].int() for i in range(1 if self.k_wl == 1 else 2, self.k_wl + 1)]
            # node encoder with second param 0 means no k-wl
            seq_x = [self.node_encoder(x_, i + 1) for i, x_ in enumerate(seq_x)]
            seq_edge_index = [batched_data[f'edge_index_{i}'].long() for i in
                              range(1 if self.k_wl == 1 else 2, self.k_wl + 1)]
            seq_edge_attr = [batched_data[f'edge_attr_{i}'].int() for i in
                             range(1 if self.k_wl == 1 else 2, self.k_wl + 1)]
            seq_assignment_index = [batched_data[f'assignment_index_{i}'].long() for i in
                                    range(1 if self.k_wl == 1 else 2, self.k_wl + 1)]
            k_wl_layers = k_wl_sequential_layers(self.num_layer, self.k_wl)
            seq_batch = [batched_data.batch]
            if 'batch_1' in batched_data:
                seq_batch.append(batched_data.batch_1)
            if 'batch_2' in batched_data:
                seq_batch.append(batched_data.batch_2)
            if 'batch_3' in batched_data:
                seq_batch.append(batched_data.batch_3)

            k_wl_h = []
        if self.gnn_type == "gat":
            edge_attr = self.edge_encoder(edge_attr)
        ### computing input node embedding

        # Only works for integer features!!!
        if self.gnn_type == "gin":
            x = x.long()
        edge_attr = edge_attr.long()
        h_list = [self.node_encoder(x, 0)]
        current_k_wl = 0
        h = h_list[0]
        for layer in range(self.num_layer):
            if layer in k_wl_layers:
                current_k_wl += 1
                k_wl_h.append(scatter_mean(h, seq_batch[k_wl_layers.index(layer)], dim=0))
                h = avg_pool_custom(h_list[k_wl_layers[0] - 1], seq_assignment_index[k_wl_layers.index(layer)])
                h = torch.cat([h, seq_x[k_wl_layers.index(layer)]], dim=1)
            h = self.convs[layer](h,
                                  edge_index if current_k_wl == 0 else seq_edge_index[current_k_wl - 1],
                                  edge_attr if current_k_wl == 0 else seq_edge_attr[current_k_wl - 1])
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
                if self.sequential_k_wl:
                    k_wl_h.append(scatter_mean(h, seq_batch[-1], dim=0))
                    h = torch.cat(k_wl_h, dim=1)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last", residual=False, gnn_type='gin',
                 node_encoder=lambda x: x, edge_encoder=lambda x: x, len_type=0):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.len_type = len_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        self.gnn_type = gnn_type
        self.edge_encoder = edge_encoder

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_encoder))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_encoder))
            elif gnn_type == 'gat':
                self.convs.append(GATv2Conv(in_channels=emb_dim, out_channels=emb_dim, edge_dim=1))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                    torch.nn.ReLU(), \
                                    torch.nn.Linear(2 * emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                    torch.nn.ReLU()))

    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # Only works for integer features!!!
        x = x.long()
        edge_attr = edge_attr.long()

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                      self.drop_ratio, training=self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
