import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class PadNodeAttr(BaseTransform):
    r"""To make it easy to run GNNs without node encoders, this pads node features to a given length by addding zeros.

    Args:
        node_attr_size (int): Target length of node features AFTER applying this transformation
    """
    def __init__(self, node_attr_size: int = True):
        assert node_attr_size > 0
        self.node_attr_size = node_attr_size

    def __call__(self, data: Data) -> Data:
        data.x = F.pad(data.x, (0, self.node_attr_size - data.x.shape[1]), "constant", 0)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(node_attr_size={self.node_attr_size})')