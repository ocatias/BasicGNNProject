import torch
from torch import cat
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class AddZeroEdgeAttr(BaseTransform):
    r"""To make it easy to run GNNs that expect edge_attr on graphs without them, this graph transformation gives every edge a zero edge feature / attribute.
        Ideally, the size of the attributes is set to the embedding dimension and the edge encoder to the identity function.

    Args:
        edge_attr_size (int): Length of the attributes that will be added to each edge
    """

    def __init__(self, edge_attr_size: int = True):
        self.edge_attr_size = edge_attr_size

    def __call__(self, data: Data) -> Data:
        if data.edge_attr is not None:
            if self.edge_attr_size == 0 and data.edge_attr.shape[1] > 0:
                return data
            data['edge_attr'] = cat(
                [torch.zeros((data.edge_index.shape[1], max(self.edge_attr_size, 1))), data.edge_attr],
                dim=1).long()
        else:
            data['edge_attr'] = torch.zeros((data.edge_index.shape[1], max(self.edge_attr_size, 1)))

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(s={self.edge_attr_size})')


class DebugTransform(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data: Data) -> Data:
        print(data)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')


class AddZeroNodeAttr(BaseTransform):
    r"""To mesake it easy to run GNNs that expect node attributes as x on graphs without them, this graph transformation gives every node a zero node feature / attribute.
        Ideally, the size of the attribute is set to the embedding dimension and the node encoder to the identity function.

    Args:
        node_attr_size (int): Length of the attributes that will be added to each edge
    """

    def __init__(self, node_attr_size: int = True):
        self.node_attr_size = node_attr_size

    def __call__(self, data: Data) -> Data:
        if data.x is not None:
            # only add attribute ib there are none
            if self.node_attr_size == 0 and data.x.shape[1] > 0:
                return data
            data['x'] = cat([torch.zeros((data.num_nodes, max(self.node_attr_size, 1))), data.x], dim=1).long()
        else:
            data['x'] = torch.zeros((data.num_nodes, max(self.node_attr_size, 1)))
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(s={self.node_attr_size})')
