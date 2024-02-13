"""
Create a train / val / test split for a dataset.
Stores the split in the directory specified in the config.
"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import QM9

from Misc.config import config


seed = 42

def main():
    dir = os.path.join(config.DATA_PATH, "QM9", "Original")
    dataset = QM9(root=dir)
    X = np.arange(len(dataset))
    y = np.zeros([len(dataset)])
    
    print(X.shape)
    print(y.shape)
    
    train_index, remaining_index, _, _ = train_test_split(X, y, test_size=0.20, random_state=seed)
    val_index, test_index, _, _ = train_test_split(remaining_index, y[:len(remaining_index)], test_size=0.50, random_state=seed)
    print(train_index.shape)
    print(val_index.shape)
    print(test_index.shape)
    
    indices_dir_path = os.path.join(config.SPLITS_PATH, "QM9")
    
    if not os.path.isdir(indices_dir_path):
        os.mkdir(indices_dir_path)
        
    to_str_list = lambda ls: [str(x) for x in ls]
        
    with open(os.path.join(indices_dir_path, "train.index"), "w") as file:
        file.write(", ".join(to_str_list(train_index)))
    with open(os.path.join(indices_dir_path, "val.index"), "w") as file:
        file.write(", ".join(to_str_list(val_index)))
    with open(os.path.join(indices_dir_path, "test.index"), "w") as file:
        file.write(", ".join(to_str_list(test_index)))

if __name__ == "__main__":
    main()


quit()


tt_path = os.path.join(DATA_PATH, "Train_Test_Splits")
tvt_path = os.path.join(DATA_PATH, "Train_Val_Test_Splits")
if not os.path.isdir(tt_path):
    os.mkdir(tt_path)
if not os.path.isdir(tvt_path):
    os.mkdir(tvt_path)


for dataset_name in datasets:
    dataset = torch_geometric.datasets.TUDataset(root=DATA_PATH, name=dataset_name)
    for fold in range(folds):
        train_index, test_index = cv_split(dataset, seed, fold, folds)

        # Train and Test splits
        with open(os.path.join(tt_path, f"{dataset_name}_fold_{fold}_of_{folds}_train.json"), "w") as file:
            json.dump(list(train_index.tolist()), file)
        with open(os.path.join(tt_path, f"{dataset_name}_fold_{fold}_of_{folds}_test.json"), "w") as file:
            json.dump(list(test_index.tolist()), file)

        # Indices to data so we can split again
        training_index_y = [(idx, dataset[idx].y) for idx in train_index]

        # Train, Val and Test splits
        train_index, val_index = stratified_data_split(training_index_y, seed)

        with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_train.json"), "w") as file:
            json.dump(list(train_index), file)
        with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_valid.json"), "w") as file:
            json.dump(list(val_index), file)
        with open(os.path.join(tvt_path, f"{dataset_name}_fold_{fold}_of_{folds}_test.json"), "w") as file:
            json.dump(list(test_index.tolist()), file)