import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.options.display.width = 0
df = pd.read_csv('project_my.csv')
print(df)
print(df.columns)
df_filtered = df
df_filtered = df_filtered.dropna(subset=['cross_val_id'])
df_filtered['test_metric'] = df_filtered['Final/Test/accuracy'].fillna(df_filtered['Final/Test/rocauc (ogb)'])
df_filtered['val_metric'] = df_filtered['Final/Val/accuracy'].fillna(df_filtered['Final/Val/rocauc (ogb)'])

print(df_filtered[df_filtered['dataset'] == 'ogbg-moltox21'])

df_grouped = df_filtered.groupby('cross_val_id').agg(
    {'dataset': 'first', 'test_metric': ['mean', 'std'], 'val_metric': ['mean', 'std'],
     'transform_k_wl': ['first', 'count'], 'k_wl_pool_function': 'first',
     'emb_dim': 'first', 'sequential_k_wl': 'first', 'k_wl_set_based': 'first',
     'connected_k_wl_last_k': 'first', 'k_wl_turbo': 'first', 'add_num_triangles': 'first', '_runtime': 'mean', 'k_wl_compute_attributes':'first'})
df_grouped.columns = ['_'.join(col) for col in df_grouped.columns.values]
print(df_grouped.columns)
print(df_grouped)
df_grouped = df_grouped.sort_values('test_metric_mean').reset_index(drop=True)
df_grouped = df_grouped[df_grouped['test_metric_mean'] != None]
df_grouped = df_grouped.dropna(subset=['test_metric_mean'])
# df_grouped = df_grouped[df_grouped['transform_k_wl_count'] ==5]
df_grouped_2 = df_grouped.groupby(
    ['dataset_first', 'k_wl_compute_attributes_first', 'k_wl_turbo_first', 'transform_k_wl_first', 'sequential_k_wl_first',
     'connected_k_wl_last_k_first']) \
    .agg({
    'test_metric_mean': ['mean'], 'val_metric_mean': ['mean', 'count'],
    '_runtime_mean': ['mean'],

})
# print(df_grouped)
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(df_grouped_2)
