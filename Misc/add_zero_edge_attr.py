import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class AddZeroEdgeAttr(BaseTransform):
    r"""To make it easy to run GNNs that expect edge_attr on graphs without them, this graph transformation gives every edge a zero edge feature / attribute.
        Ideally, the size of the attributes is set to the embedding dimension and the edge encoder to the identity function.

    Args:
        edge_attr_size (int): Length of the attributes that will be added to each edge
    """

    def __init__(self, edge_attr_size: int = True):
        assert edge_attr_size > 0
        self.edge_attr_size = edge_attr_size

    def __call__(self, data: Data) -> Data:
        data.edge_attr = torch.zeros((data.edge_index.shape[1], self.edge_attr_size))
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(edge_attr_size={self.edge_attr_size})')

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
        assert node_attr_size > 0
        self.node_attr_size = node_attr_size

    def __call__(self, data: Data) -> Data:
        if 'x' not in data.keys:
            data['x'] = torch.zeros((data.num_nodes, self.node_attr_size))
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(node_attr_size={self.node_attr_size})')
