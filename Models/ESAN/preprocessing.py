"""
Adapted from https://github.com/beabevi/ESAN (MIT LICENSE)
"""

from typing import Optional, Union, Tuple
from collections import defaultdict

import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import coalesce

ORIG_EDGE_INDEX_KEY = 'original_edge_index'

class SubgraphData(Data):
    def __inc__(self, key, value, store):
        if key == ORIG_EDGE_INDEX_KEY:
            return self.num_nodes_per_subgraph
        else:
            return super().__inc__(key, value)
        
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "subgraph_idx":
            return 0
        else:
           return super().__cat_dim__(key, value, *args, **kwargs)


# TODO: update Pytorch Geometric since this function is on the newest version
def to_undirected(edge_index: Tensor, edge_attr: Optional[Tensor] = None,
                  num_nodes: Optional[int] = None,
                  reduce: str = "add") -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features. (default: :obj:`"add"`)
    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :class:`Tensor`)
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    if edge_attr is not None:
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes, reduce)

    if edge_attr is None:
        return edge_index
    else:
        return edge_index, edge_attr


def preprocess(dataset, transform):
    def unbatch_subgraphs(data):
        subgraphs = []
        num_nodes = data.num_nodes_per_subgraph.item()
        for i in range(data.num_subgraphs):
            edge_index, edge_attr = subgraph(torch.arange(num_nodes) + (i * num_nodes),
                                             data.edge_index, data.edge_attr,
                                             relabel_nodes=False, num_nodes=data.x.size(0))
            subgraphs.append(
                Data(
                    x=data.x[i * num_nodes: (i + 1) * num_nodes, :], edge_index=edge_index - (i * num_nodes),
                    edge_attr=edge_attr,
                    subgraph_idx=torch.tensor(0), subgraph_node_idx=torch.arange(num_nodes),
                    num_nodes=num_nodes,
                )
            )

        original_edge_attr = data.original_edge_attr if data.edge_attr is not None else data.edge_attr
        return Data(x=subgraphs[0].x, edge_index=data.original_edge_index, edge_attr=original_edge_attr, y=data.y,
                    subgraphs=subgraphs)

    data_list = [unbatch_subgraphs(data) for data in dataset]

    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)
    dataset.transform = transform
    return dataset


class Graph2Subgraph:
    def __init__(self, process_subgraphs=lambda x: x, pbar=None):
        self.process_subgraphs = process_subgraphs
        self.pbar = pbar

    def __call__(self, data):
        assert data.is_undirected()

        subgraphs = self.to_subgraphs(data)
        subgraphs = [self.process_subgraphs(s) for s in subgraphs]

        batch = Batch.from_data_list(subgraphs)

        if self.pbar is not None: next(self.pbar)

        return SubgraphData(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                            subgraph_batch=batch.batch,
                            y=data.y, subgraph_idx=batch.subgraph_idx, subgraph_node_idx=batch.subgraph_node_idx,
                            num_subgraphs=len(subgraphs), num_nodes_per_subgraph=data.num_nodes,
                            original_edge_index=data.edge_index, original_edge_attr=data.edge_attr)

    def to_subgraphs(self, data):
        raise NotImplementedError


class EdgeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data):
        # remove one of the bidirectional index
        if data.edge_attr is not None and len(data.edge_attr.shape) == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)

        keep_edge = data.edge_index[0] <= data.edge_index[1]
        edge_index = data.edge_index[:, keep_edge]
        edge_attr = data.edge_attr[keep_edge, :] if data.edge_attr is not None else data.edge_attr

        subgraphs = []

        for i in range(edge_index.size(1)):
            subgraph_edge_index = torch.hstack([edge_index[:, :i], edge_index[:, i + 1:]])
            subgraph_edge_attr = torch.vstack([edge_attr[:i], edge_attr[i + 1:]]) \
                if data.edge_attr is not None else data.edge_attr

            if data.edge_attr is not None:
                subgraph_edge_index, subgraph_edge_attr = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                                        num_nodes=data.num_nodes)
            else:
                subgraph_edge_index = to_undirected(subgraph_edge_index, subgraph_edge_attr,
                                                    num_nodes=data.num_nodes)

            subgraphs.append(
                Data(
                    x=data.x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        if len(subgraphs) == 0:
            subgraphs = [
                Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                     subgraph_idx=torch.tensor(0), subgraph_node_idx=torch.arange(data.num_nodes),
                     num_nodes=data.num_nodes,
                     )
            ]
        return subgraphs
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class NodeDeleted(Graph2Subgraph):
    def to_subgraphs(self, data):
        subgraphs = []
        all_nodes = torch.arange(data.num_nodes)

        for i in range(data.num_nodes):
            subset = torch.cat([all_nodes[:i], all_nodes[i + 1:]])
            subgraph_edge_index, subgraph_edge_attr = subgraph(subset, data.edge_index, data.edge_attr,
                                                               relabel_nodes=False, num_nodes=data.num_nodes)

            subgraphs.append(
                Data(
                    x=data.x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class EgoNets(Graph2Subgraph):
    def __init__(self, num_hops, add_node_idx=False, process_subgraphs=lambda x: x, pbar=None):
        super().__init__(process_subgraphs, pbar)
        self.num_hops = num_hops
        self.add_node_idx = add_node_idx

    def to_subgraphs(self, data):

        subgraphs = []

        for i in range(data.num_nodes):

            _, _, _, edge_mask = k_hop_subgraph(i, self.num_hops, data.edge_index, relabel_nodes=False,
                                                num_nodes=data.num_nodes)
            subgraph_edge_index = data.edge_index[:, edge_mask]
            subgraph_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else data.edge_attr

            x = data.x
            if self.add_node_idx:
                # prepend a feature [0, 1] for all non-central nodes
                # a feature [1, 0] for the central node
                ids = torch.arange(2).repeat(data.num_nodes, 1)
                ids[i] = torch.tensor([ids[i, 1], ids[i, 0]])

                x = torch.hstack([ids, data.x]) if data.x is not None else ids.to(torch.float)

            subgraphs.append(
                Data(
                    x=x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr,
                    subgraph_idx=torch.tensor(i), subgraph_node_idx=torch.arange(data.num_nodes),
                    num_nodes=data.num_nodes,
                )
            )
        return subgraphs

    # Todo: maybe add process_subgraphs and pbar
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.num_hops)}, {self.add_node_idx})'

def policy2transform(policy: str, num_hops, process_subgraphs=lambda x: x, pbar=None):
    if policy == "edge_deleted":
        return EdgeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "node_deleted":
        return NodeDeleted(process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets":
        return EgoNets(num_hops, process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "ego_nets_plus":
        return EgoNets(num_hops, add_node_idx=True, process_subgraphs=process_subgraphs, pbar=pbar)
    elif policy == "original":
        return process_subgraphs

    raise ValueError("Invalid subgraph policy type")