import ast

import pandas as pd
from torch_geometric.datasets import TUDataset

from Misc.biconnected_components_finder import BiconnectedComponents
from Misc.graph_visualizations import visualize
from Misc.transform_to_k_wl import create_adjacency_from_graph, get_number_of_triangles

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.width = 0
df = pd.read_csv('project_original.csv')

print(df.columns)
df_filtered = df
df_filtered = df_filtered.dropna(subset=['predicted_list'])
df_filtered = df_filtered.loc[df_filtered['dataset'] == 'IMDB-BINARY']

df_filtered.added_num_triangles = df_filtered.added_num_triangles.fillna(False)
df_filtered.k_wl_turbo = df_filtered.k_wl_turbo.fillna('False')
df_filtered.k_wl_turbo_version = df_filtered.k_wl_turbo_version.fillna('False')

architectures = set(df_filtered['architecture'].values.tolist())
results = dict()
for arch in architectures:
    for tri in [True, False]:
        for turbo in ['False', 'disconnected', 'both_k_disconnected']:
            r = list(map(ast.literal_eval, df_filtered[(df_filtered.architecture == arch) &
                                                       (df_filtered.k_wl_turbo_version == turbo) &
                                                       (df_filtered.added_num_triangles == tri)
                                                       ].correct_list.values.tolist()))
            if len(r) == 0:
                continue
            results[f'{arch}_{tri}_{turbo}'] = [round(x / len(r), 2) for x in list(map(sum, zip(*r)))]
print(len(results))


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= 70
dataset = TUDataset('data/IMDB', name='IMDB-BINARY', pre_filter=MyFilter())
header = sorted(results.keys())
print(header)
for i, k in enumerate(
        zip(*[results[x] for x in header])):
    if sum(k) == len(results):
        continue

    g = dataset.get(i)

    bf = BiconnectedComponents(g)
    groups = bf.BCC()

    num_triangles = get_number_of_triangles(g)
    print(i, len(groups), num_triangles, k)
    # visualize(g, f'{i}_{str(k)}')
