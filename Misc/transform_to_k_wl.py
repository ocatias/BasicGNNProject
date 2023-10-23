import copy
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from itertools import product, combinations
from math import factorial, comb
from os import path
from pprint import pprint
from statistics import mean

import networkx as nx
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch import transpose, stack, mode, tensor, cat, zeros, empty, int32, int8
from torch.nn.functional import pad
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_scipy_sparse_matrix

from Misc.biconnected_components_finder import BiconnectedComponents
from Misc.graph_visualizations import visualize


def get_number_of_triangles(adj):
    count = 0
    n = len(adj)
    for i in range(n):
        for j in range(n):
            if (adj[i][j] == 1):
                for k in range(n):
                    if (adj[j][k] == 1 and adj[k][i] == 1):
                        count += 1

    return count // 6


def graph_from_adj(adj):
    graph = from_scipy_sparse_matrix(adj)
    return Data(x=tensor([0] * adj.shape[0]), edge_index=graph[0], edge_attr=graph[1].long(), y=tensor([1]))


def create_adjacency_from_graph(graph):
    adj = [[None for j in range(graph.num_nodes)] for i in range(graph.num_nodes)]
    if graph.edge_attr is not None:
        attrs = graph['edge_attr'].tolist()
    else:
        attrs = defaultdict(lambda: 1)
    for i, x in enumerate(transpose(graph['edge_index'], 0, 1)):
        adj[x[0]][x[1]] = attrs[i]
    return adj


class TransforToKWl(BaseTransform):
    def __init__(self, k: int, turbo=False, max_group_size=40, agg_function_features: str = 'cat'):
        if not 2 <= k <= 3:
            raise NotImplementedError('k-WL: k can be only 2 or 3 at the moment')
        self.k = k
        if __name__ == '__main__':
            self.graph_data_path = path.join('..', 'metadata', 'k_wl_graphs')
        else:
            self.graph_data_path = path.join('metadata', 'k_wl_graphs')
        self.max_group_size = max_group_size
        self.range_k = list(range(k))
        self.matrices = {}
        self.uses_turbo = turbo
        self.average_num_of_vertices = 0
        self.average_num_of_new_vertices = 0
        self.vertices_num = defaultdict(int)
        self.vertices_reduction = defaultdict(lambda: defaultdict(int))
        self.k_wl_vertices_num = defaultdict(int)
        self.stats_isomorphism_indexes = []
        self.stats_triangle_counts = []
        self.processed_num = 0
        self.agg_function_features_name = agg_function_features
        if agg_function_features is None or agg_function_features == 'mode':
            self.agg_function_features = lambda x: mode(stack(x), dim=0).values
            self.num_edge_repeat = 1
        elif agg_function_features == 'cat':
            # there is always only 1 different vertex in vertex groups for new vertexes.
            # So we have k + 1 choose 2 possible different edges that can melt into one (in a fully connected subgraph)
            self.num_edge_repeat = comb(self.k + 1, 2)
            self.agg_function_features = self.safe_cat
        self.nan_tensor_edge_features = None

    def safe_cat(self, selected_attrs: list, edge=True):
        if edge and len(selected_attrs) < self.num_edge_repeat:
            for _ in range(self.num_edge_repeat - len(selected_attrs)):
                selected_attrs.append(self.nan_tensor_edge_features)

        return cat(selected_attrs)

    def create_empty_graph(self, n):
        if n == 0:
            return [], [[]]
        saved_graph = self.load_empty_graph(n)
        if saved_graph is not None:
            return saved_graph
        all_combinations = list(product(list(range(n)), repeat=self.k))
        # all_combinations = list(combinations(list(range(n)), self.k))
        edges = [[], []]
        edge_attributes = []
        for i, c1 in enumerate(all_combinations):
            for j, c2 in enumerate(all_combinations):
                # the adjacency is simple. If the new vertices share old vertices except one.
                # The number is the position where they differ
                common = self.has_common(c1, c2)
                if common is not None:
                    edges[0].append(i)
                    edges[1].append(j)
                    edge_attributes.append(common)

        if len(all_combinations) != n ** self.k:
            raise ValueError('numbers dont add up', len(all_combinations), n, self.k)
        # TODO: add check for number of edges
        self.save_empty_graph((all_combinations, edges, edge_attributes), n)
        return all_combinations, edges, edge_attributes

    def has_common(self, c1, c2):
        diff_num = 0
        diff_pos = None
        for i in self.range_k:
            if c1[i] != c2[i]:
                diff_num += 1
                # just to have 0 to mark edges that are not changed in k-kl turbo.
                diff_pos = i + 1
        if diff_num != 1:
            return None
        return diff_pos + 1

    def graph_to_k_wl_graph(self, graph, return_mapping=None):
        vert_num = graph.num_nodes
        num_edges = graph.edge_index.shape[1]
        self.k_wl_vertices_num[vert_num] += 1
        if graph.edge_attr is not None:
            if len(graph.edge_attr.shape) == 1:
                len_edge_attr = 1
            else:
                len_edge_attr = graph.edge_attr.shape[1]
        else:
            len_edge_attr = 0
        self.nan_tensor_edge_features = tensor([float('NaN')] * len_edge_attr)
        if graph.x is not None:
            if len(graph.x.shape) == 1:
                len_vert_attr = 1
            else:
                len_vert_attr = graph.x.shape[1]
        else:
            len_vert_attr = 0
        if vert_num < 30:
            if vert_num not in self.matrices:
                self.matrices[vert_num] = self.create_empty_graph(vert_num)
            all_combinations, new_edges, new_edge_attr = deepcopy(self.matrices[vert_num])
        else:
            all_combinations, new_edges, new_edge_attr = self.create_empty_graph(vert_num)

        old_adj = create_adjacency_from_graph(graph)
        new_x = [0] * len(all_combinations)
        if len_edge_attr > 0:
            for i in range(len(new_edge_attr)):
                c1 = all_combinations[new_edges[0][i]]
                c2 = all_combinations[new_edges[1][i]]
                # edge attributes are hard. This will include median of all the edges that
                # were between any of the vertexes from the two subgraphs. It can be remade to just include the stack.
                # The first attribute is the number generated by the k_WL algorithm form the function has_common
                # TODO: try stacking the features
                selected_attrs = [tensor(old_adj[x[0]][x[1]]).resize(len_edge_attr) for x in
                                  combinations(set(c1 + c2), 2) if
                                  old_adj[x[0]][x[1]] is not None]
                # in case there was no edge between any of the graph vertices
                if len(selected_attrs) == 0:

                    new_edge_attr[i] = cat((tensor([new_edge_attr[i]]),
                                            zeros((len_edge_attr * self.num_edge_repeat),
                                                  dtype=int32)))
                else:
                    # for x in selected_attrs:
                    #     print(x)
                    #     print(x.shape)
                    # print(selected_attrs)
                    new_edge_attr[i] = cat((tensor([new_edge_attr[i]]),
                                            self.agg_function_features(selected_attrs, True)))
                    # print('new_edge_attr[i]', new_edge_attr[i])
                # elif len(selected_attrs) == 1:
                #
                #
                #     if len_edge_attr > 1:
                #         new_edge_attr[i] = cat((tensor([new_edge_attr[i]]),
                #                                 selected_attrs[0],
                #                                 ))
                #     else:
                #         new_edge_attr[i] = cat((tensor([new_edge_attr[i]]), tensor(selected_attrs)))
                #
                # else:
                #     if len_edge_attr > 1:
                #         new_edge_attr[i] = cat((tensor([new_edge_attr[i]]),
                #                                 mode(stack(selected_attrs),
                #                                      dim=0).values))
                #     else:
                #         new_edge_attr[i] = cat((tensor([new_edge_attr[i]]),
                #                                 mode(stack(selected_attrs)).values.reshape(1)))

        else:
            new_edge_attr = [tensor([i]) for i in new_edge_attr]
        self.stats_isomorphism_indexes.append(defaultdict(int))
        for i, c in enumerate(all_combinations):
            # TODO: try keeping the order of vertices for the isomorphism test
            # Old version
            # works only for K==2 and K==3
            # sum of number of edges in the subgraph plus 1
            # for K larger than 3, I would suggest using hash from WL algorithm on each small subgraph
            # Using bool to detect where edge has value and where None is.
            # k_x = [sum([bool(old_adj[c[j - 1]][c[j]]) for j in range(len(c))]) + 1]

            # New test version. Keeping in mind the order of vertices. Each binary place represents one edge
            k_x = [sum([int(bool(old_adj[c[j - 1]][c[j]])) * 2 ** j for j in range(len(c))]) + 1]
            self.stats_isomorphism_indexes[-1][k_x[0]] += 1
            # adding all vertex features from the vertex in the subgraph using mode to keep the dimensionality.
            if len_vert_attr > 0:
                new_x[i] = cat(
                    (tensor(k_x), self.agg_function_features([graph.x[j].reshape(len_vert_attr) for j in c], False),),
                    0)
            # elif len_vert_attr == 1:
            #     new_x[i] = cat((tensor(k_x), mode(stack([graph.x[j] for j in c])).values.reshape(1)))
            else:
                new_x[i] = tensor(k_x)

        graph.x = stack(new_x)
        graph.num_nodes = len(all_combinations)

        if len(new_edges[0]) > 0:
            # print('new_edge_attr', new_edge_attr)
            graph.edge_attr = stack(new_edge_attr)
            graph.edge_index = tensor(new_edges)
        else:
            graph.edge_attr = empty((0, len_edge_attr + 1), dtype=int32)
            graph.edge_index = empty((2, 0), dtype=int32)
        self.average_num_of_vertices = mean((self.average_num_of_vertices, vert_num))
        self.average_num_of_new_vertices = mean((self.average_num_of_new_vertices, vert_num ** self.k))
        if return_mapping is not None:
            if not isinstance(return_mapping, list):
                mapping = all_combinations
            else:
                mapping = [tuple([return_mapping[j] for j in i]) for i in all_combinations]
            return graph, mapping
        # TODO sanity check whether the graph is undirected. Check it without edge attributes
        return graph

    def __call__(self, data: Data) -> Data:
        self.stats_triangle_counts.append(get_number_of_triangles(create_adjacency_from_graph(data)))
        self.processed_num += 1
        if self.processed_num % 100 == 0:
            print(f'transform to k-WL -- done {self.processed_num}')
        num_nodes = data.num_nodes
        self.vertices_num[num_nodes] += 1
        if num_nodes < 2 or data.edge_index.shape[1] == 0:
            return self.add_dimensions_to_graph_without_modifying(data)
        if self.uses_turbo:
            return self.k_wl_turbo(data)
        else:
            return self.graph_to_k_wl_graph(data)

    def save_empty_graph(self, graph, n):
        pth = path.join(self.graph_data_path, f'empty_graph_{self.k}_{n}.pkl')
        with open(pth, 'wb+') as file:
            pickle.dump(graph, file)

    def load_empty_graph(self, n):
        pth = path.join(self.graph_data_path, f'empty_graph_{self.k}_{n}.pkl')
        if path.exists(pth):
            with open(pth, 'rb+') as file:
                return pickle.load(file)
        else:
            return None

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k})(turbo={self.uses_turbo})'
                f'(max_group_size={self.max_group_size})(feature_pooling={self.agg_function_features_name})')

    def __del__(self):
        print('number of vertices in graphs', self.vertices_num)
        print('number of vertices processed by k-WL', self.k_wl_vertices_num)
        print('number of graphs reduced to what', self.vertices_reduction)
        print('average_num_of_vertices', self.average_num_of_vertices)
        print('average_num_of_new_vertices', self.average_num_of_new_vertices)
        print('number of triangles and isomorphism:', list(zip(self.stats_triangle_counts, self.stats_isomorphism_indexes)))

    def get_subgraph(self, graph, vertices):
        new_graph = copy.deepcopy(graph)
        new_graph.num_nodes = len(vertices)
        new_graph.x = []
        if graph.edge_attr is not None:
            new_graph.edge_attr = []
        new_graph.edge_index = ([], [])
        vertices_map = defaultdict(lambda: None)
        v_counter = 0
        for i in range(graph.num_nodes):
            if i in vertices:
                vertices_map[i] = v_counter
                v_counter += 1
                new_graph.x.append(graph.x[i])

        for i in range(graph.edge_index.shape[1]):
            if int(graph.edge_index[0][i]) in vertices and int(graph.edge_index[1][i]) in vertices:
                new_graph.edge_index[0].append(vertices_map[int(graph.edge_index[0][i])])
                new_graph.edge_index[1].append(vertices_map[int(graph.edge_index[1][i])])
                if graph.edge_attr is not None:
                    new_graph.edge_attr.append(graph.edge_attr[i])

        new_graph.x = stack(new_graph.x)
        if graph.edge_attr is not None:
            new_graph.edge_attr = stack(new_graph.edge_attr)
        new_graph.edge_index = tensor(new_graph.edge_index)
        return new_graph

    @staticmethod
    def save_picture_of_graph(graph, name):
        plt.clf()
        plt.figure(figsize=(18, 18))
        g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
        nx.draw_networkx(g)
        plt.savefig(f'../pictures/graph_{name}.png')

    def k_wl_turbo(self, graph):
        bf = BiconnectedComponents(graph)
        groups = bf.BCC()
        groups = [x for x in groups if len(x) <= self.max_group_size or self.max_group_size == -1]
        vertices_in_components = defaultdict(lambda: False)
        old_vertex_to_group_mapping = dict()
        for j, g in enumerate(groups):
            for i in g:
                vertices_in_components[i] = True
                old_vertex_to_group_mapping[i] = j
        old_vertex_not_in_subgraphs = [i for i in range(graph.num_nodes) if not vertices_in_components[i]]
        new_graph = copy.deepcopy(graph)
        # each of the subgraph will have n^k vertices and all vertices that will not be changed
        new_graph.num_nodes = sum([len(n) ** self.k for n in groups]) \
                              + graph.num_nodes - sum(vertices_in_components.values())

        # get all subgraphs detected by BCC and convert them using k-WL algorithm
        processed_subgraphs = [self.graph_to_k_wl_graph(self.get_subgraph(graph, g), g) for g in groups]
        # create mapping between vertices of new graph and vertices in old graph. Left side is old vertex index.
        # Only for vertices not in any subgraph!
        old_vertex_to_new_vertex_mapping_no_subgraps = {o: i for i, o in enumerate(old_vertex_not_in_subgraphs)}
        # inverted version of the mapping
        simple_vertex_mapping_inv = {v: k for k, v in old_vertex_to_new_vertex_mapping_no_subgraps.items()}

        group_to_new_vertex_mapping = dict()
        new_vertex_to_group_mapping = dict()
        mapped_vertices = len(old_vertex_to_new_vertex_mapping_no_subgraps)
        for i, g in enumerate(groups):
            group_to_new_vertex_mapping[i] = list(range(mapped_vertices, mapped_vertices + len(g) ** self.k))
            mapped_vertices += len(group_to_new_vertex_mapping[i])
            for k in group_to_new_vertex_mapping[i]:
                new_vertex_to_group_mapping[k] = i

        new_graph.x = []
        new_graph.edge_attr = []
        new_graph.edge_index = ([], [])

        # add simple elements of graph first
        for v in old_vertex_not_in_subgraphs:
            # add padding to x and edge attr because the subgraphs added later have one dimension more
            new_graph.x.append(pad(graph.x[v], pad=(1, 0), value=0))
        for i, (x, y) in enumerate(zip(*graph.edge_index.tolist())):
            if x in old_vertex_not_in_subgraphs and y in old_vertex_not_in_subgraphs:
                new_graph.edge_attr.append(pad(graph.edge_attr[i], pad=(1, 0), value=0))
                new_graph.edge_index[0].append(tensor(old_vertex_to_new_vertex_mapping_no_subgraps[x]))
                new_graph.edge_index[1].append(tensor(old_vertex_to_new_vertex_mapping_no_subgraps[y]))

        # add subgraphs one by one with all their connections
        # skipping connections to vertices not yet in graph
        # in case there is a connection between two subgraphs.
        # If a connection apperas between a vertice and a subgraph,
        # the vertice will be connected to all the vertices of the subgraph.
        # In case of subgraph to subgraph connection, all vertices of one subgraph
        # will be connected to all vertices of the other one
        already_processed_old_vertices = copy.copy(old_vertex_not_in_subgraphs)
        for sub_i, (subgraph, subgraph_mapping) in enumerate(processed_subgraphs):
            # adding vertexes and intra subgraph edges
            starting_id = len(new_graph.x)
            new_graph.x.extend(list(subgraph.x))
            new_graph.edge_attr.extend(list(subgraph.edge_attr))
            for i, j in zip(*subgraph.edge_index):
                new_graph.edge_index[0].append(i + starting_id)
                new_graph.edge_index[1].append(j + starting_id)
            # adding edges between subgraph and already present vertices
            # adding edges only between the k-tuples where the original vertice is present
            #  and not to all the new vertices in the subgraph
            for e_i, (i, j) in enumerate(zip(*graph.edge_index)):
                i = int(i)
                j = int(j)
                if i in groups[sub_i] and j in already_processed_old_vertices:
                    if j in old_vertex_not_in_subgraphs:
                        for i_v in range(starting_id, starting_id + subgraph.num_nodes):
                            if i in subgraph_mapping[i_v - starting_id]:
                                new_graph.edge_attr.append(pad(graph.edge_attr[e_i], pad=(1, 0), value=0))
                                new_graph.edge_index[0].append(i_v)
                                new_graph.edge_index[1].append(old_vertex_to_new_vertex_mapping_no_subgraps[j])
                    else:
                        other_group_starting_new_vertex_id = \
                            group_to_new_vertex_mapping[old_vertex_to_group_mapping[j]][0]
                        other_group_old_vertex_mapping = processed_subgraphs[old_vertex_to_group_mapping[j]][1]
                        size_of_other_group = len(group_to_new_vertex_mapping[old_vertex_to_group_mapping[j]])
                        for i_v in range(starting_id, starting_id + subgraph.num_nodes):
                            for j_v in range(other_group_starting_new_vertex_id,
                                             other_group_starting_new_vertex_id + size_of_other_group):

                                if i in subgraph_mapping[i_v - starting_id] and \
                                        j in other_group_old_vertex_mapping[j_v - other_group_starting_new_vertex_id]:
                                    new_graph.edge_attr.append(pad(graph.edge_attr[e_i], pad=(1, 0), value=0))
                                    new_graph.edge_index[0].append(i_v)
                                    new_graph.edge_index[1].append(j_v)
                # other way
                elif j in groups[sub_i] and i in already_processed_old_vertices:
                    if i in old_vertex_not_in_subgraphs:
                        for j_v in range(starting_id, starting_id + subgraph.num_nodes):
                            if j in subgraph_mapping[j_v - starting_id]:
                                new_graph.edge_attr.append(pad(graph.edge_attr[e_i], pad=(1, 0), value=0))
                                new_graph.edge_index[0].append(old_vertex_to_new_vertex_mapping_no_subgraps[i])
                                new_graph.edge_index[1].append(j_v)
                    else:
                        other_group_starting_new_vertex_id = \
                            group_to_new_vertex_mapping[old_vertex_to_group_mapping[i]][0]
                        size_of_other_group = len(group_to_new_vertex_mapping[old_vertex_to_group_mapping[i]])
                        other_group_old_vertex_mapping = processed_subgraphs[old_vertex_to_group_mapping[i]][1]
                        for j_v in range(starting_id, starting_id + subgraph.num_nodes):
                            for i_v in range(other_group_starting_new_vertex_id,
                                             other_group_starting_new_vertex_id + size_of_other_group):

                                if j in subgraph_mapping[j_v - starting_id] and \
                                        i in other_group_old_vertex_mapping[i_v - other_group_starting_new_vertex_id]:
                                    new_graph.edge_attr.append(pad(graph.edge_attr[e_i], pad=(1, 0), value=0))
                                    new_graph.edge_index[0].append(i_v)
                                    new_graph.edge_index[1].append(j_v)
            already_processed_old_vertices.extend(groups[sub_i])
        new_graph.x = stack(new_graph.x)
        new_graph.edge_attr = stack(new_graph.edge_attr)
        new_graph.edge_index = tensor(new_graph.edge_index)
        return new_graph

    def add_dimensions_to_graph_without_modifying(self, data):
        if data.edge_attr.shape[0] == 0:
            data.edge_attr = empty((0, data.edge_attr.shape[1] + 1), dtype=int8)
        else:
            data.edge_attr = pad(data.edge_attr, pad=(1, 0, 0, 0), value=0)
        data.x = pad(data.x, pad=(1, 0, 0, 0), value=0)
        if self.agg_function_features_name == 'cat':
            data.edge_attr = pad(data.edge_attr,
                                 pad=(0, (self.num_edge_repeat - 1) * (data.edge_attr.shape[1] - 1)),
                                 value=0)
            data.x = pad(data.x,
                         pad=(0, (self.k - 1) * (data.x.shape[1] - 1)),
                         value=0)
        return data


if __name__ == '__main__':
    with open('../debug/graph_20_21.pkl', 'rb') as file:
        data = pickle.load(file)
    transform = TransforToKWl(3)

    from sknetwork.data import house

    # data = transform.graph_from_adj(house())
    visualize(data, 'transformed_before')  # , labels=[0, 1, 2, 3, 4], figsize=(3, 3))
    transformed_data, mapping = transform.graph_to_k_wl_graph(data, True)
    # print(transformed_data.edge_attr)
    # pprint(list(zip(mapping, transformed_data.x)))
    # visualize(transformed_data, 'transformed_k=2', labels=mapping, e_feat_dim=0)

# TODO try on zinc dataset
# TODO try concatenating features instead of mode
# TODO try setting up embeddings for

# TODO CSL dataset look at triange counts
