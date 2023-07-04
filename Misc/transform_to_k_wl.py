from copy import deepcopy
from itertools import product, combinations
from statistics import mean

import torch
from torch import transpose, stack, mode, tensor, cat, zeros, empty, int32
from torch.nn.functional import pad
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class TransforToKWl(BaseTransform):
    def __init__(self, k: int = True):
        if not 2 <= k <= 3:
            raise NotImplementedError('k-WL: k can be only 2 or 3 at the moment')
        self.k = k
        self.matrices = {}
        for k in range(30):
            self.matrices[k] = (self.create_empty_matrix(k))

        self.average_num_of_vertices = 0
        self.average_num_of_new_vertices = 0

    def create_empty_matrix(self, n):
        if n == 0:
            return [], [[]]
        all_combinations = list(product(list(range(n)), repeat=self.k))
        # all_combinations = list(combinations(list(range(n)), self.k))
        new_adj = [[None for j in range(len(all_combinations))] for i in range(len(all_combinations))]
        for i, c1 in enumerate(all_combinations):
            for j, c2 in enumerate(all_combinations):
                # the adjacency is simple. If the new vertices share old vertices except one.
                # The number is the position where they differ
                new_adj[i][j] = self.has_common(c1, c2)
        if len(all_combinations) != n ** self.k:
            raise ValueError('numbers dont add up', len(all_combinations), n, self.k)
        return all_combinations, new_adj

    def has_common(self, c1, c2):
        diff_num = 0
        for i in range(len(c1)):
            if c1[i] != c2[i]:
                diff_num += 1
        if diff_num != 1:
            return None

        for i in range(len(c1)):
            if c1[i] != c2[i]:
                return i + 1

    def create_adjacency_from_graph(self, graph, size):
        adj = [[None for j in range(size)] for i in range(size)]
        attrs = graph['edge_attr'].tolist()
        for i, x in enumerate(transpose(graph['edge_index'], 0, 1)):
            adj[x[0]][x[1]] = attrs[i]
        return adj

    def graph_to_k_wl_graph(self, graph):
        vert_num = graph['num_nodes']
        num_edges = graph.edge_attr.shape[0]
        if vert_num < 2 or num_edges == 0:
            if num_edges == 0:
                graph.edge_attr = empty((0, graph.edge_attr.shape[1] + 1), dtype=int32)
            else:
                graph.edge_attr = graph.edge_attr.expand(-1, graph.edge_attr.shape[1] + 1)
            graph.x = pad(graph.x, pad=(1, 0, 0, 0), value=0)
            return graph
        len_edge_attr = graph.edge_attr.shape[1]
        if vert_num < 40:
            if vert_num not in self.matrices:
                self.matrices[vert_num] = self.create_empty_matrix(vert_num)
            all_combinations, new_adj = deepcopy(self.matrices[vert_num])
        else:
            all_combinations, new_adj = self.create_empty_matrix(vert_num)

        old_adj = self.create_adjacency_from_graph(graph, vert_num)
        new_x = [0] * len(new_adj)
        for i, c1 in enumerate(all_combinations):
            for j, c2 in enumerate(all_combinations):
                # edge attributes are hard. This will include median of all the edges that
                # were between any of the vertexes from the two subgraphs. It can be remade to just include the stack.
                # The first attribute is the number generated by the k_WL algorithm form the function has_common
                if new_adj[i][j] is not None:
                    selected_attrs = [tensor(old_adj[x[0]][x[1]]) for x in
                                      combinations(set(c1 + c2), 2) if
                                      old_adj[x[0]][x[1]] is not None]

                    # in case there was no edge between any of the graph vertices
                    if len(selected_attrs) == 0:
                        new_adj[i][j] = cat((tensor([new_adj[i][j]]),
                                             zeros((len_edge_attr), dtype=int32)))
                    else:
                        new_adj[i][j] = cat((tensor([new_adj[i][j]]),
                                             mode(stack(selected_attrs),
                                                  dim=0).values))

        for i, c in enumerate(all_combinations):
            # works only for K==2 and K==3
            # sum of number of edges in the subgraph
            # for K larger than 3, I would suggest using hash from WL algorithm on each small subgraph
            # Using bool to detect where edge has value and where None is.
            k_x = [sum([bool(old_adj[c[j - 1]][c[j]]) for j in range(len(c))])]
            # adding all vertex features from the vertex in the subgraph using mode to keep the dimensionality.
            new_x[i] = cat((tensor(k_x), mode(stack([graph.x[j] for j in c]), dim=0).values), 0)

        graph.x = stack(new_x)
        graph.num_nodes = len(all_combinations)
        new_edge = [[], []]
        new_edge_attr = []
        # transform to Torch graph data
        for i, x in enumerate(new_adj):
            for j, e in enumerate(x):
                if e is not None:
                    new_edge[0].append(i)
                    new_edge[1].append(j)
                    new_edge_attr.append(e)

        if len(new_edge[0]) > 0:
            graph.edge_attr = stack(new_edge_attr)
            graph.edge_index = tensor(new_edge)
        else:
            graph.edge_attr = empty((0, len_edge_attr + 1), dtype=int32)
            graph.edge_index = empty((2, 0), dtype=int32)
        self.average_num_of_vertices = mean((self.average_num_of_vertices, vert_num))
        self.average_num_of_new_vertices = mean((self.average_num_of_new_vertices, vert_num ** self.k))
        return graph

    def __call__(self, data: Data) -> Data:
        return self.graph_to_k_wl_graph(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k})')

    def __del__(self):
        print('average_num_of_vertices', self.average_num_of_vertices)
        print('average_num_of_new_vertices', self.average_num_of_new_vertices)