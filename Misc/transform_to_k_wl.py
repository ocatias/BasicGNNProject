import copy
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from itertools import product, combinations
from math import factorial, comb, ceil
from os import path
from statistics import mean
from uuid import uuid4

import networkx as nx
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch import transpose, stack, mode, tensor, cat, zeros, empty, int32, int8, unique, empty_like
from torch.nn.functional import pad
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_scipy_sparse_matrix

from Misc.biconnected_components_finder import BiconnectedComponents
from Misc.graph_visualizations import visualize
from Models.utils import device


def edge_in_same_group(groups, simple_v, i, j):
    for g in groups:
        if i in g and j in g:
            return True
    if i in simple_v and j in simple_v:
        return True
    return False


def k_wl_sequential_layers(n, k):
    if k == 1:
        k = 2
    d = ceil(n / k)
    o = list(range(d, n, d))[:k - 1]
    if len(o) < k - 1:
        o.append(n - 1)
    return o


def get_number_of_triangles_per_node(graph):
    g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    tri = nx.triangles(g).values()
    return tri


def get_number_of_triangles(graph):
    g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    tri = sum(nx.triangles(g).values()) / 3
    return tri
    count = 0
    adj = create_adjacency_from_graph(graph)
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
    return Data(x=tensor([0] * adj.shape[0]), edge_index=graph[0], edge_attr=graph[1].long(),
                y=tensor([1], device=device()))


def create_adjacency_from_graph(graph):
    adj = [[None for j in range(graph.num_nodes)] for i in range(graph.num_nodes)]
    if graph.edge_attr is not None:
        attrs = graph['edge_attr'].tolist()
    else:
        attrs = defaultdict(lambda: 1)
    for i, x in enumerate(transpose(graph['edge_index'], 0, 1)):
        adj[x[0]][x[1]] = attrs[i]
    return adj


def sort_tuple(t):
    return tuple(sorted(t))


def mapping_to_assignment_index(m, offset=0):
    k = len(m[0])
    out = [[-1] * (k * len(m)), [-1] * (k * len(m))]
    for i, c in enumerate(m):
        for j, e in enumerate(c):
            out[0][i * k + j] = e
            out[1][i * k + j] = i + offset
    return out
    # return {i: c for i, c in enumerate(m)}


def get_subgraph(graph, vertices):
    new_graph = type(graph)()
    new_graph['num_nodes'] = len(vertices)
    x = []
    if graph.edge_attr is not None:
        edge_attr = []
    edge_index = ([], [])
    vertices_map = defaultdict(lambda: None)
    v_counter = 0
    for i in range(graph.num_nodes):
        if i in vertices:
            vertices_map[i] = v_counter
            v_counter += 1
            if graph.x is not None:
                x.append(graph.x[i])

    for i in range(graph.edge_index.shape[1]):
        if int(graph.edge_index[0][i]) in vertices and int(graph.edge_index[1][i]) in vertices:
            edge_index[0].append(vertices_map[int(graph.edge_index[0][i])])
            edge_index[1].append(vertices_map[int(graph.edge_index[1][i])])
            if graph.edge_attr is not None:
                edge_attr.append(graph.edge_attr[i])

    if len(x) > 0:
        new_graph['x'] = stack(x)
    if graph.edge_attr is not None:
        new_graph['edge_attr'] = stack(edge_attr)
    new_graph['edge_index'] = tensor(edge_index, device=device())
    return new_graph


class TransforToKWl(BaseTransform):
    def __init__(self, k: int, turbo=False, max_group_size=500, agg_function_features: str = 'cat', set_based=False,
                 modify=False, connected=False, super_connected=False, compute_attributes=True):
        # if not 2 <= k <= 3:
        #     raise NotImplementedError('k-WL: k can be only 2 or 3 at the moment')
        self.k = k
        self.set_based = set_based
        self.modify = modify
        self.connected = connected
        self.super_connected = super_connected
        self.compute_attributes = compute_attributes
        if __name__ == '__main__':
            self.graph_data_path = path.join('..', 'metadata', 'k_wl_graphs')
        else:
            self.graph_data_path = path.join('metadata', 'k_wl_graphs')
        self.max_group_size = max_group_size
        self.range_k = list(range(k))
        self.matrices = {}
        self.uses_turbo = turbo
        if self.uses_turbo and self.modify:
            raise ValueError('k-wl turbo and modify (not using sequantial) is not compatible')
        self.average_num_of_vertices = 0
        self.average_num_of_new_vertices = 0
        self.vertices_num = defaultdict(int)
        self.vertices_reduction = defaultdict(lambda: defaultdict(int))
        self.k_wl_vertices_num = defaultdict(int)
        self.stats_isomorphism_indexes = defaultdict(int)
        self.stats_triangle_counts = []
        self.processed_num = 0
        self.agg_function_features_name = agg_function_features
        if agg_function_features is None or agg_function_features == 'mode':
            self.agg_function_features = lambda x, y: tensor(mode(stack(x), dim=0).values, device=device())
            self.num_edge_repeat = 1
        elif agg_function_features == 'cat':
            # there is always only 1 different vertex in vertex groups for new vertexes.
            # So we have k + 1 choose 2 possible different edges that can melt into one (in a fully connected subgraph)
            self.num_edge_repeat = comb(self.k + 1, 2)
            self.agg_function_features = self.safe_cat
        else:
            raise ValueError('agg_function_features must be mode or cat', agg_function_features)
        self.nan_tensor_edge_features = None
        self.last_processed_data = []

    def safe_cat(self, selected_attrs: list, edge=True):
        if edge and len(selected_attrs) < self.num_edge_repeat:
            for _ in range(self.num_edge_repeat - len(selected_attrs)):
                selected_attrs.append(self.nan_tensor_edge_features)

        return cat(selected_attrs)

    def create_empty_graph_set(self, graph):
        adj = create_adjacency_from_graph(graph)
        n = len(adj)
        if n == 0:
            return [], [[]]
        all_combinations = list(combinations(list(range(n)), self.k))
        if self.connected:
            all_combinations = [c for c in all_combinations if
                                sum([bool(adj[c[j - 1]][c[j]]) for j in range(len(c))]) > (
                                        (0 if self.k == 2 else 1) + (1 if self.super_connected else 0))]
        combinations_index = {c: i for i, c in enumerate(all_combinations)}
        edges = [[], []]
        edge_attributes = []
        used_edges = set()
        for i, e in enumerate(zip(*graph.edge_index.tolist())):
            e = sort_tuple(e)
            if e in used_edges:
                continue
            else:
                used_edges.add(e)
            l = [x for x in list(range(n)) if x not in set(e)]
            for x in list(combinations(l, self.k - 1)):
                t1 = sort_tuple(((e[0],) + x))
                t2 = sort_tuple(((e[1],) + x))
                if self.connected and (t1 not in combinations_index.keys() or t2 not in combinations_index.keys()):
                    continue

                i_0 = combinations_index[t1]
                i_1 = combinations_index[t2]
                edges[0].append(i_0)
                edges[1].append(i_1)
                edge_attributes.append(1)
                edges[0].append(i_1)
                edges[1].append(i_0)
                edge_attributes.append(1)
        return all_combinations, edges, edge_attributes

    def create_empty_graph_tuple(self, graph):

        adj = create_adjacency_from_graph(graph)
        n = len(adj)
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

    def has_common_edge_2(self, c1, c2, adj):
        if len(set(c1).intersection(set(c2))) == self.k - 1:
            diff = list(set(c1).symmetric_difference(set(c2)))
            if adj[diff[0]][diff[1]] is not None:
                return 1
            else:
                return None
        else:
            return None

    def has_common_edge(self, c1, c2, adj):
        # check this
        if len(set(c1).union(set(c2))) == self.k + 1:
            s = 0
            for i in c1:
                for j in c2:
                    if i != j:
                        if adj[i][j] is not None:
                            s += 1
            if s == 0:
                return None
            else:
                return s
        else:
            return None

    def has_common_no_order(self, c1, c2):
        #  v2 only return if they have an edge in the original graph between c1 and c2
        if len(set(c1).union(set(c2))) == self.k + 1:
            return 1
        else:
            return None

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

    def graph_to_k_wl_graph(self, graph, return_mapping=None, override_modify=None):
        self.last_processed_data.append(graph)
        if override_modify is not None:
            local_modify = override_modify
        else:
            local_modify = self.modify
        if self.uses_turbo and self.k == 1:
            if graph.x is None:
                graph.x = tensor([1] * graph.num_nodes, device=device())
            else:
                if isinstance(graph.x, list):
                    graph.x = tensor(graph.x, device=device())
                graph.x = pad(graph.x, pad=(0, 1), value=1)
            graph['iso_type_1'] = tensor([1] * graph.num_nodes, device=device())
            graph['edge_index_1'] = graph.edge_index
            graph['edge_attr_1'] = graph.edge_attr
            return graph, [(x,) for x in return_mapping]
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
        self.nan_tensor_edge_features = tensor([float('NaN')] * len_edge_attr, device=device())
        if graph.x is not None:
            if len(graph.x.shape) == 1:
                len_vert_attr = 1
            else:
                len_vert_attr = graph.x.shape[1]
        else:
            len_vert_attr = 0
        # if vert_num < 30:
        #     if vert_num not in self.matrices:
        #         self.matrices[vert_num] = self.create_empty_graph(vert_num)
        #     all_combinations, new_edges, new_edge_attr = deepcopy(self.matrices[vert_num])
        # else:
        old_adj = create_adjacency_from_graph(graph)
        if self.set_based:
            all_combinations, new_edges, new_edge_attr = self.create_empty_graph_set(graph)
        else:
            all_combinations, new_edges, new_edge_attr = self.create_empty_graph_tuple(graph)
        new_x = [0] * len(all_combinations)
        if self.compute_attributes and len_edge_attr > 0:
            for i in range(len(new_edge_attr)):
                c1 = all_combinations[new_edges[0][i]]
                c2 = all_combinations[new_edges[1][i]]
                if self.set_based:
                    e = set(c1) ^ set(c2)
                    if self.compute_attributes:
                        new_edge_attr[i] = tensor(old_adj[e.pop()][e.pop()], device=device())
                    else:
                        new_edge_attr[i] = tensor([new_edge_attr[i]], device=device())
                else:
                    # edge attributes are hard. This will include median of all the edges that
                    # were between any of the vertexes from the two subgraphs. It can be remade to just include the stack.
                    # The first attribute is the number generated by the k_WL algorithm form the function has_common
                    selected_attrs = [tensor(old_adj[x[0]][x[1]], device=device()).resize(len_edge_attr) for x in
                                      combinations(set(c1 + c2), 2) if
                                      old_adj[x[0]][x[1]] is not None]
                    # in case there was no edge between any of the graph vertices
                    if len(selected_attrs) == 0:
                        new_edge_attr[i] = cat((tensor([new_edge_attr[i]], device=device()),
                                                zeros((len_edge_attr * self.num_edge_repeat),
                                                      dtype=int32, device=device())))
                    else:
                        new_edge_attr[i] = cat((tensor([new_edge_attr[i]], device=device()),
                                                self.agg_function_features(selected_attrs, True)))

        else:
            new_edge_attr = [tensor([i], device=device()) for i in new_edge_attr]
        for i, c in enumerate(all_combinations):
            if self.set_based:
                # Set version
                # works only for K==2 and K==3
                # sum of number of edges in the subgraph plus 1
                # for K larger than 3, I would suggest using hash from WL algorithm on each small subgraph
                # Using bool to detect where edge has value and where None is.
                k_x = [sum([bool(old_adj[c[j - 1]][c[j]]) for j in range(len(c))]) + 1]
            else:
                # Tuple version. Keeping in mind the order of vertices. Each binary place represents one edge
                k_x = [sum([int(bool(old_adj[c[j - 1]][c[j]])) * 2 ** j for j in range(len(c))]) + 1]
            self.stats_isomorphism_indexes[k_x[0]] += 1
            # adding all vertex features from the vertex in the subgraph using mode to keep the dimensionality.
            if self.compute_attributes and len_vert_attr > 0:
                if len(new_x) == 0:
                    print('graph', graph)
                    print(all_combinations)

                new_x[i] = cat(
                    (tensor(k_x, device=device()),
                     self.agg_function_features([graph.x[j].reshape(len_vert_attr) for j in c], False),),
                    0)
            else:
                new_x[i] = tensor(k_x, device=device()).long()

        graph['iso_type' + ("" if local_modify else f"_{self.k}")] = stack(new_x).squeeze(dim=1)

        if 'x' in graph and len(graph.x.shape) == 1:
            graph['x'] = torch.unsqueeze(graph.x, 1)
        if len(graph['iso_type' + ("" if local_modify else f"_{self.k}")].shape) == 1:
            graph['iso_type' + ("" if local_modify else f"_{self.k}")] = torch.unsqueeze(
                graph['iso_type' + ("" if local_modify else f"_{self.k}")], 1)
        if local_modify:
            graph.num_nodes = len(all_combinations)

        if len(new_edges[0]) > 0:
            new_edge_attr = stack(new_edge_attr)
            new_edge_index = tensor(new_edges, device=device())
        else:
            new_edge_attr = empty((0, len_edge_attr if self.set_based else len_edge_attr + 1), dtype=int32,
                                  device=device())
            new_edge_index = empty((2, 0), dtype=int32, device=device())
        graph[f'assignment_index_{self.k}'] = tensor(mapping_to_assignment_index(all_combinations), device=device())
        graph['edge_index' + ("" if local_modify else f"_{self.k}")] = new_edge_index
        graph['edge_attr' + ("" if local_modify else f"_{self.k}")] = new_edge_attr

        # if 'x' not in graph.keys:
        #     graph['x'] = tensor([[1.0]] * graph.num_nodes, device=device())
        # if 'edge_attr' not in graph.keys:
        #     graph['edge_attr'] = tensor([[1.0]] * num_edges, device=device())
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
        self.last_processed_data = []
        self.stats_triangle_counts.append(get_number_of_triangles(data))
        self.processed_num += 1
        if self.processed_num % 100 == 0:
            print(f'transform to k-WL -- done {self.processed_num}')
        num_nodes = data.num_nodes
        self.vertices_num[num_nodes] += 1
        if self.uses_turbo:
            return self.k_wl_turbo(data)
        if num_nodes <= self.k or data.edge_index.shape[1] == 0:
            return self.add_dimensions_to_graph_without_modifying(data)
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
        return (f'{self.__class__.__name__}(k={self.k})(tur={self.uses_turbo})'
                f'(fp={self.agg_function_features_name})'
                f'(set={self.set_based})(con={self.connected})(att={self.compute_attributes})')

    # def __del__(self):
    #     print('k', self.k)
    #     print('number of vertices in graphs', self.vertices_num)
    #     print('number of vertices processed by k-WL', self.k_wl_vertices_num)
    #     print('number of graphs reduced to what', self.vertices_reduction)
    #     print('average_num_of_vertices', self.average_num_of_vertices)
    #     print('average_num_of_new_vertices', self.average_num_of_new_vertices)
    #     print('triangle counts', self.stats_triangle_counts)
    #     print('last processed data', self.last_processed_data)
    #     print('isomorphisms distribution:', self.stats_isomorphism_indexes)

    # with open(path.join('Results', f'isomorphism_{self.k}_{str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))}.txt'),
    #           'wt') as f:
    #     f.writelines([str(x) for x in zip(self.stats_triangle_counts, self.stats_isomorphism_indexes)])

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

        # get all subgraphs detected by BCC and convert them using k-WL algorithm
        processed_subgraphs = [self.graph_to_k_wl_graph(get_subgraph(graph, g), g, override_modify=False) for g in
                               groups]
        # create mapping between vertices of new graph and vertices in old graph. Left side is old vertex index.
        # Only for vertices not in any subgraph!
        old_vertex_to_new_vertex_mapping_no_subgraps = {o: i for i, o in enumerate(old_vertex_not_in_subgraphs)}
        # inverted version of the mapping
        simple_vertex_mapping_inv = {v: k for k, v in old_vertex_to_new_vertex_mapping_no_subgraps.items()}

        group_to_new_vertex_mapping = dict()
        new_vertex_to_group_mapping = dict()
        mapped_vertices = len(old_vertex_to_new_vertex_mapping_no_subgraps)
        for i, g in enumerate(groups):
            group_to_new_vertex_mapping[i] = \
                list(range(mapped_vertices, mapped_vertices + processed_subgraphs[i][0][f'iso_type_{self.k}'].shape[0]))
            mapped_vertices += len(group_to_new_vertex_mapping[i])
            for k in group_to_new_vertex_mapping[i]:
                new_vertex_to_group_mapping[k] = i

        new_graph.num_nodes = sum([g[f'iso_type_{self.k}'].shape[0] for g, _ in processed_subgraphs]) \
                              + graph.num_nodes - sum(vertices_in_components.values())
        num_edges = sum([g[f'edge_index_{self.k}'].shape[1] for g, _ in processed_subgraphs])
        new_graph['x'] = empty(
            size=(new_graph.num_nodes, 1 + (graph.x.shape[1] if self.compute_attributes and 'x' in graph else 0)),
            device=device())
        if self.compute_attributes and 'edge_attr' in graph:
            new_graph['edge_attr'] = empty(size=(num_edges, graph.edge_attr.shape[1]), device=device())
        new_graph['edge_index'] = empty(size=(2, num_edges), device=device(), dtype=torch.int)
        assignment_index = [[-1] * len(old_vertex_not_in_subgraphs),
                            [-1] * len(old_vertex_not_in_subgraphs)]

        cum_i_node = 0
        cum_i_edge = 0
        # add simple elements of graph first
        for i, v in enumerate(old_vertex_not_in_subgraphs):
            if self.compute_attributes and graph.x is not None:
                # add padding to x and edge attr because the subgraphs added later have one dimension more,
                # or if the compute_attributes is 'cat', the subgraphs have one dimension for isomorphism
                # class at the beginning and k times the attributes.
                if self.agg_function_features_name == 'cat':
                    att = pad(graph.x[v], pad=(1, 0), value=0)
                    for _ in range(self.k - 1):
                        att = cat((att, graph.x[v]))
                    new_graph.x[i] = att
                else:
                    new_graph.x[i] = pad(graph.x[v], pad=(1, 0), value=0)
            else:
                new_graph.x[i] = 0
            assignment_index[0][i] = v
            assignment_index[1][i] = i
            cum_i_node += 1
        for _, m in processed_subgraphs:
            m_i = mapping_to_assignment_index(m, offset=max(assignment_index[1], default=-1) + 1)
            assignment_index[0].extend(m_i[0])
            assignment_index[1].extend(m_i[1])
        new_graph[f'assignment_index_{self.k}'] = tensor(assignment_index, device=device())

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
            if len(subgraph[f'iso_type_{self.k}'].shape) == 1:
                subgraph[f'iso_type_{self.k}'] = torch.unsqueeze(subgraph[f'iso_type_{self.k}'], 1)
            new_graph.x[cum_i_node:cum_i_node + subgraph[f'iso_type_{self.k}'].shape[0]] = subgraph[
                f'iso_type_{self.k}']
            if self.compute_attributes and f'edge_attr_{self.k}' in subgraph and 'edge_attr' in graph:
                new_graph.edge_attr[cum_i_edge:cum_i_edge + subgraph[f'edge_attr_{self.k}'].shape[0]] = subgraph[
                    f'edge_attr_{self.k}']
            subgraph[f'edge_index_{self.k}'] += cum_i_node
            new_graph.edge_index[:, cum_i_edge:cum_i_edge + subgraph[f'edge_index_{self.k}'].shape[1]] = subgraph[
                f'edge_index_{self.k}']
            cum_i_edge += subgraph[f'edge_index_{self.k}'].shape[1]
            cum_i_node += subgraph[f'iso_type_{self.k}'].shape[0]

            already_processed_old_vertices.extend(groups[sub_i])
        if cum_i_edge != new_graph.edge_index.shape[1]:
            print(new_graph.edge_index)
            print(cum_i_edge)
            print(new_graph.edge_index.shape)
            raise ValueError('cum_i_edge must be the same as edge index length')
        if cum_i_node != new_graph.x.shape[0]:
            print(new_graph.x)
            print(cum_i_node)
            print(new_graph.x.shape)
            raise ValueError('cum_i_node must be the same as x index length')
        if len(new_graph.x.shape) == 1:
            new_graph['x'] = torch.unsqueeze(new_graph.x, 1)
        if not new_graph.x.shape[0] == max(new_graph[f'assignment_index_{self.k}'][1]) + 1:
            print(new_graph.x.shape[0])
            print('assignment max', max(new_graph[f'assignment_index_{self.k}'][1]))
            print('edge max', torch.max(new_graph.edge_index))
            print('count distinct in assignment index', len(set(new_graph[f'assignment_index_{self.k}'][1].tolist())))
            print('distinct in assignment index', set(new_graph[f'assignment_index_{self.k}'][1].tolist()))
            print(new_graph)
            print(graph)
            print(groups)
            print(new_graph[f"assignment_index_{self.k}"])
            raise ValueError(f'values dont match {new_graph.x.shape, max(new_graph[f"assignment_index_{self.k}"][1])}')

        if self.compute_attributes and new_graph.edge_attr is not None:
            # new_graph.edge_attr = stack(new_graph.edge_attr)
            pass
        else:
            new_graph.edge_attr = empty((new_graph.edge_index.shape[1], 0), device=device())
        if self.modify:
            return new_graph
        else:
            graph[f'iso_type_{self.k}'] = new_graph.x
            graph[f'assignment_index_{self.k}'] = new_graph[f'assignment_index_{self.k}']

            graph[f'edge_attr_{self.k}'] = new_graph.edge_attr
            graph[f'edge_index_{self.k}'] = new_graph.edge_index
            return graph

    def add_dimensions_to_graph_without_modifying(self, data):
        # TODO check this whole function
        if not self.set_based:
            if data.edge_attr.shape[0] == 0:
                data['edge_attr' + ("" if self.modify else f"_{self.k}")] = empty((0, data.edge_attr.shape[1] + 1),
                                                                                  dtype=int8, device=device())
            else:
                data['edge_attr' + ("" if self.modify else f"_{self.k}")] = pad(data.edge_attr, pad=(1, 0, 0, 0),
                                                                                value=0)
        if not self.modify:
            data[f"edge_attr_{self.k}"] = zeros((data.edge_index.shape[1], data.edge_attr.shape[1]), dtype=int8,
                                                device=device())
        if self.compute_attributes and data.x is not None:
            if self.agg_function_features_name == 'cat':
                data['x' if self.modify else f"iso_type_{self.k}"] = cat([data.x] * self.k, dim=1)
            data['x' if self.modify else f"iso_type_{self.k}"] = pad(data.x, pad=(1, 0, 0, 0), value=0)
        else:
            data['x' if self.modify else f"iso_type_{self.k}"] = zeros((data.num_nodes, 1), device=device())
        if not self.modify:
            data['edge_index' + f"_{self.k}"] = data['edge_index']
        data['assignment_index' + f"_{self.k}"] = tensor([list(range(data.num_nodes)), list(range(data.num_nodes))],
                                                         device=device())
        if self.agg_function_features_name == 'cat' and not self.uses_turbo and not self.set_based:
            data['edge_attr' + ("" if self.modify else f"_{self.k}")] = pad(data.edge_attr,
                                                                            pad=(0, (self.num_edge_repeat - 1) * (
                                                                                    data.edge_attr.shape[1] - 1)),
                                                                            value=0)
        return data


if __name__ == '__main__':
    with open('../debug/graph_20_02.pkl', 'rb') as file:
        data = pickle.load(file)
    transform = TransforToKWl(3, set_based=True, turbo=False, modify=False, compute_attributes=False)

    from sknetwork.data import house

    print(data)
    # data = graph_from_adj(house())
    visualize(data, 'transformed_before')  # , labels=[0, 1, 2, 3, 4], figsize=(3, 3))
    transformed_data = transform(data)
    print(transformed_data)
    print(transformed_data.x)
    # print('unique iso type', unique(transformed_data.iso_type_3[:, 0]))
    print(transformed_data.iso_type_3)
    print(transformed_data.edge_attr)
    print(transformed_data.edge_attr_3)
    # print(transformed_data.edge_attr)
    # pprint(list(zip(mapping, transformed_data.x)))
    visualize(transformed_data, 'transformed_turbo')
