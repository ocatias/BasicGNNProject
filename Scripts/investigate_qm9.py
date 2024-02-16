import os
from tqdm import tqdm

import torch
from torch_geometric.datasets import QM9

from Misc.config import config

def main():
    dir = os.path.join(config.DATA_PATH, "QM9", "Original")
    dataset = QM9(root=dir)
    print(dataset)
    
    nr_features_x = dataset[0].x.shape[1]
    nr_features_e = dataset[0].edge_attr.shape[1]
    print(dataset[0])
    print(dataset[0].x.shape)
    max_values_x = [0 for _ in range(nr_features_x)]
    max_values_e = [0 for _ in range(nr_features_e)]
    
    for graph in tqdm(dataset):
        for i in range(nr_features_x):
            max_value = torch.max(graph.x[:, i])
            if max_value > max_values_x[i]:
                max_values_x[i] = int(max_value.item())
        for i in range(nr_features_e):
            max_value = torch.max(graph.edge_attr[:, i])
            if max_value > max_values_e[i]:
                max_values_e[i] = int(max_value.item())  
    print(f"Vertex features: {max_values_x}")
    print(f"Edge features: {max_values_e}")
    

if __name__ == "__main__":
    main()