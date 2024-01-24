from torch import cat, tensor, unsqueeze
from torch.nn.functional import pad
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from Models.utils import device


class AddNodeDegree(BaseTransform):

    def __init__(self):
        pass

    def __call__(self, data: Data) -> Data:
        l = [[0] for _ in range(data.num_nodes)]
        for i in data.edge_index[0]:
            l[i][0] += 1
        if data.x is not None:
            data['x'] = cat([ tensor(l, device=device()), data.x], dim=1).long()
        else:
            data['x'] = tensor(l, device=device()).long()
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
