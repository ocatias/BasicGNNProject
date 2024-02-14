"""
Adapted from https://github.com/beabevi/ESAN (MIT LICENSE)
"""

import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention

from Models.utils import get_pooling_fct

def subgraph_pool(h_node, batched_data, pool):
    # Represent each subgraph as the pool of its node representations
    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
                     torch.cumsum(num_subgraphs, dim=0)])
    graph_offset = tmp[batched_data.batch]

    # Original
    # subgraph_idx = graph_offset + batched_data.subgraph_batch

    # Changed:
    # ToDo: Investigate whether this causes problems
    subgraph_idx = batched_data.subgraph_batch
    
    return pool(h_node, subgraph_idx)

class DSnetwork(torch.nn.Module):
    def __init__(self, subgraph_gnn, subgraph_gnn_out_dim, channels, num_tasks, invariant, subgraph_pool):
        super(DSnetwork, self).__init__()
        self.subgraph_gnn = subgraph_gnn
        self.invariant = invariant
        self.num_tasks = num_tasks
        self.subgraph_pool = get_pooling_fct(subgraph_pool)
        
        fc_list = []
        fc_sum_list = []
        for i in range(len(channels)):
            fc_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn_out_dim,
                                           out_features=channels[i]))
            if self.invariant:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i],
                                                   out_features=channels[i]))
            else:
                fc_sum_list.append(torch.nn.Linear(in_features=channels[i - 1] if i > 0 else subgraph_gnn_out_dim,
                                                   out_features=channels[i]))

        self.fc_list = torch.nn.ModuleList(fc_list)
        self.fc_sum_list = torch.nn.ModuleList(fc_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=channels[-1], out_features=2 * channels[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * channels[-1], out_features=num_tasks)
        )

    def forward(self, batched_data):
        h_node = self.subgraph_gnn(batched_data)
        h_subgraph = subgraph_pool(h_node, batched_data, self.subgraph_pool)

        if self.invariant:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)

                h_subgraph = F.elu(x1)

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
            for layer_idx, fc_sum in enumerate(self.fc_sum_list):
                h_graph = F.elu(fc_sum(h_graph))
        else:
            for layer_idx, (fc, fc_sum) in enumerate(zip(self.fc_list, self.fc_sum_list)):
                x1 = fc(h_subgraph)
                x2 = fc_sum(
                    torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")
                )

                h_subgraph = F.elu(x1 + x2[batched_data.subgraph_idx_batch])

            # aggregate to obtain a representation of the graph given the representations of the subgraphs
            h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph)


class DSSnetwork(torch.nn.Module):
    def __init__(self, num_layers, in_dim, emb_dim, num_tasks, feature_encoder, GNNConv, edge_encoder):
        super(DSSnetwork, self).__init__()

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.feature_encoder = feature_encoder
        self.edge_encoder = edge_encoder

        gnn_list = []
        gnn_sum_list = []
        bn_list = []
        bn_sum_list = []
        for i in range(num_layers):
            gnn_list.append(GNNConv(emb_dim))
            bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            gnn_sum_list.append(GNNConv(emb_dim))
            bn_sum_list.append(torch.nn.BatchNorm1d(emb_dim))

        self.gnn_list = torch.nn.ModuleList(gnn_list)
        self.gnn_sum_list = torch.nn.ModuleList(gnn_sum_list)

        self.bn_list = torch.nn.ModuleList(bn_list)
        self.bn_sum_list = torch.nn.ModuleList(bn_sum_list)

        self.final_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim, out_features=2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2 * emb_dim, out_features=num_tasks)
        )

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        original_edge_attr = batched_data.original_edge_attr
        x = self.feature_encoder(x)
        
        if batched_data.original_edge_attr is not None and len(batched_data.original_edge_attr.shape) == 1:
            original_edge_attr = original_edge_attr.view(-1, 1)
        
        edge_attr = self.edge_encoder(edge_attr)
        original_edge_attr = self.edge_encoder(original_edge_attr)
        for i in range(len(self.gnn_list)):
            
            gnn, bn, gnn_sum, bn_sum = self.gnn_list[i], self.bn_list[i], self.gnn_sum_list[i], self.bn_sum_list[i]

            h1 = bn(gnn(x, edge_index, edge_attr))

            num_nodes_per_subgraph = batched_data.num_nodes_per_subgraph
            tmp = torch.cat([torch.zeros(1, device=num_nodes_per_subgraph.device, dtype=num_nodes_per_subgraph.dtype),
                             torch.cumsum(num_nodes_per_subgraph, dim=0)])
            graph_offset = tmp[batch]

            # Same idx for a node appearing in different subgraphs of the same graph
            node_idx = graph_offset + batched_data.subgraph_node_idx
            x_sum = torch_scatter.scatter(src=x, index=node_idx, dim=0, reduce="mean")

            h2 = bn_sum(gnn_sum(x_sum, batched_data.original_edge_index, original_edge_attr if edge_attr is not None else edge_attr))

            x = F.relu(h1 + h2[node_idx])

        h_subgraph = subgraph_pool(x, batched_data, global_mean_pool)
        # aggregate to obtain a representation of the graph given the representations of the subgraphs

        h_graph = torch_scatter.scatter(src=h_subgraph, index=batched_data.subgraph_idx_batch, dim=0, reduce="mean")

        return self.final_layers(h_graph)