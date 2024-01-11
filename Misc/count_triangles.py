from torch.nn.functional import pad
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from Misc.transform_to_k_wl import create_adjacency_from_graph, get_number_of_triangles


class CountTriangles(BaseTransform):

    def __init__(self):
        pass

    def __call__(self, data: Data) -> Data:
        adj = create_adjacency_from_graph(data)
        num_triangles = get_number_of_triangles(adj)
        data.x = pad(data.x, pad=(1, 0), value=num_triangles)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')