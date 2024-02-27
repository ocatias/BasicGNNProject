from collections import defaultdict

import networkx as nx
from matplotlib import pyplot as plt

from Misc.utils import edge_tensor_to_list

colors_feat = ['white', 'red', 'orange', 'yellow', 'blue', 'green', 'grey', 'pink', 'magenta']
colors_type = [ 'orange',  'blue', 'pink', 'grey', 'green', 'magenta', 'yellow', 'orange',  'blue', 'pink', 'grey', 'green', 'magenta','yellow']
cololrs_edges = ['orange', 'red', 'yellow', 'blue', 'green', 'grey', 'pink', 'purple', 'magenta', 'maroon']
default_color = 'white'
default_color_edge = 'black'



def visualize(data, name, labels=None, colors=None, v_feat_dim=None, e_feat_dim=None, figsize=(30,30)):
    G = nx.Graph(directed=False)
    plt.clf()
    plt.figure(figsize=figsize, dpi=100)
    if labels is None:
        labels = defaultdict(lambda x: x)
    elif isinstance(labels, list):
        labels = {i: v for i, v in enumerate(labels)}

    if colors is None:
        colors = colors_type
    edge_list = edge_tensor_to_list(data.edge_index)

    for i, edge in enumerate(edge_list):
        c = default_color_edge
        if (e_feat_dim is not None):
            if len(data.edge_attr.shape) == 1:
                c = cololrs_edges[data.edge_attr[i]]
            else:
                c = cololrs_edges[data.edge_attr[i, e_feat_dim]]
        G.add_edge(edge[0], edge[1],
                   color=c)

    color_map_vertices = [default_color for _ in G]
    color_map_edges = [default_color_edge for _ in range(data.edge_index.shape[1])]

    if v_feat_dim is not None:
        node_types = data.x[:, v_feat_dim].tolist()
        for i, node in enumerate(G):
            color_map_vertices[i] = colors[node_types[node]]

    edges = G.edges()
    color_map_edges = [G[u][v]['color'] for u, v in edges]

    cc= list(nx.connected_components(G))
    if len(cc) > 0 and len(cc[0]) < len(labels):
        labels = {k:v for k,v in labels.items() if k in cc[0]}
    nx.draw(G, pos=nx.kamada_kawai_layout(G),node_color=color_map_vertices, edge_color=color_map_edges, with_labels=True,
                         font_weight='bold', arrows=False, labels=labels, font_color='red')
    plt.savefig(f'../pictures/{name}.png')
    plt.close()
