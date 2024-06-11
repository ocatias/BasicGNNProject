import pandas as pd
from pandas import to_numeric

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.width = 0
df = pd.read_csv('project_original.csv')

print(df.columns)
df_filtered = df
df_filtered['difference_acc'] = df_filtered['best_test_acc'] - df_filtered['best_val_acc']
df_filtered['difference_acc'] = abs(df_filtered['difference_acc'])
df_filtered = df_filtered[df_filtered['difference_acc'] != None]
# df_filtered = df_filtered[df_filtered['difference_acc'] < 0.05]

df_filtered.added_num_triangles = df_filtered.added_num_triangles.fillna(False)
df_filtered.k_wl_turbo = df_filtered.k_wl_turbo.fillna(False)
df_filtered.k_wl_turbo_version = df_filtered.k_wl_turbo_version.fillna(False)
df_filtered.k2_wl_connected = df_filtered.k2_wl_connected.fillna(False)
df_filtered = df_filtered.fillna(False)
df_filtered = df_filtered.sort_values('best_test_acc')
# print(df_filtered[['best_test_acc', 'best_val_acc', 'architecture']])

df_grouped = df_filtered.groupby('cross_valid_id').agg(
    {'best_test_acc': ['mean', 'std'], 'best_val_acc': ['mean', 'std'],
     'architecture': ['first', 'count'], 'random_seed_set': 'first', 'dataset': 'first',
     'added_num_triangles': 'first', 'k_wl_turbo_version': 'first', 'k_wl_turbo': 'first', 'k2_wl_connected': 'first'})
df_grouped.columns = ['_'.join(col) for col in df_grouped.columns.values]
print(df_grouped.columns)
df_grouped = df_grouped.sort_values('best_test_acc_mean').reset_index(drop=True)
df_grouped = df_grouped[df_grouped['best_test_acc_mean'] != None]
df_grouped.random_seed_set_first = df_grouped.random_seed_set_first.fillna(False)
df_grouped = df_grouped[df_grouped['architecture_count'] == 10]
# df_grouped.head()
df_grouped_2 = df_grouped.groupby(
    ['dataset_first', 'random_seed_set_first', 'architecture_first',
     'added_num_triangles_first', 'k_wl_turbo_version_first','k_wl_turbo_first']) \
    .agg({'best_test_acc_mean': ['count', 'mean','std'], 'best_val_acc_mean': ['mean','std']})
# print(df_grouped)
print(df_grouped_2.reset_index())
df_grouped_2 =  df_grouped_2.reset_index()
df_grouped_2.columns = ['_'.join(col) for col in df_grouped_2.columns.values]
df_grouped_2.best_val_acc_mean_mean = to_numeric(df_grouped_2.best_val_acc_mean_mean)
df_grouped_2.best_test_acc_mean_mean = to_numeric(df_grouped_2.best_test_acc_mean_mean)
print(df_grouped_2.columns)
print(df_grouped_2.dtypes)
print(df_grouped_2)
df_grouped_2.round(4).to_csv('out_original.csv')
