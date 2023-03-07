"""
Completely ignores the graph structure and just applies an MLP to the pooled vertex features
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList
from torch_geometric.nn import global_add_pool, global_mean_pool

class MLP(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, num_tasks, dropout_rate, graph_pooling = "sum"):
        super(MLP, self).__init__()

        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        self.layers = ModuleList([Linear(num_features, hidden), ReLU()])
        for _ in range(num_layers-1):
            self.layers.append(Linear(hidden, hidden))
            self.layers.append(ReLU())

        self.final_lin = Linear(hidden, num_classes*num_tasks)
        
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("unknown pooling")

    def forward(self, data):
        x = self.pool(data.x, data.batch).float()
        for layer in self.layers:
            x = layer(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.final_lin(x)

        if self.num_tasks == 1:
            x = x.view(-1, self.num_classes)
        else:
            x.view(-1, self.num_tasks, self.num_classes)
        return x

    def __repr__(self):
        return self.__class__.__name__