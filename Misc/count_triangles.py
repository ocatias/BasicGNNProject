from torch import cat, tensor, stack
from torch.nn.functional import pad
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from Misc.transform_to_k_wl import create_adjacency_from_graph, get_number_of_triangles, \
    get_number_of_triangles_per_node
from Models.utils import device


class CountTriangles(BaseTransform):

    def __init__(self):
        pass

    def __call__(self, data: Data) -> Data:
        num_triangles = get_number_of_triangles_per_node(data)

        if data.x is not None:
            data['x'] = cat((data.x, tensor(num_triangles, device=device()).long().view((-1,1))), dim=1)
        else:
            data['x'] = tensor(num_triangles, device=device()).long().view((-1,1))
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')
