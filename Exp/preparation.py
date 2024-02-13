import os
import csv

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset, LRGBDataset, QM9
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims

from Models.mpnn import MPNN
from Models.encoder import NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder
from Models.mlp import MLP
from Misc.drop_features import DropFeatures
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
from Misc.cosine_scheduler import get_cosine_schedule_with_warmup
from Misc.select_only_one_target import SelectOnlyOneTarget

def get_transform(args):
    transforms = []
    dataset_name_lowercase = args.dataset.lower()
    if dataset_name_lowercase == "csl":
        transforms.append(OneHotDegree(5))
        
        # Pad features if necessary (needs to be done after adding additional features from other transformation)
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))
        
    # For dataset name QM9_i we only predict the i-th target value
    if "qm9" in dataset_name_lowercase and "_" in dataset_name_lowercase:
        target = int(dataset_name_lowercase.split("_")[1])
        assert target >= 0 and target <= 18
        transforms.append(SelectOnlyOneTarget(target))
         
    if args.do_drop_feat:
        transforms.append(DropFeatures(args.emb_dim))

    return Compose(transforms)

def load_indices(dataset_name, config):
    all_idx = {}
    for section in ['train', 'val', 'test']:
        with open(os.path.join(config.SPLITS_PATH, dataset_name,  f"{section}.index"), 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx

def load_dataset(args, config):
    transform = get_transform(args)
    dataset_name = args.dataset.lower()

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str)

    # ZINC
    if dataset_name == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
        
    # OGB graph level tasks
    elif dataset_name in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
        
    # Cyclic Skip Link dataset
    elif dataset_name == "csl":
        indices = load_indices("CSL", config)
        dataset = GNNBenchmarkDataset(name ="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[indices["train"][args.split]], dataset[indices["val"][args.split]], dataset[indices["test"][args.split]]]
        
    # Long Rage Graph Benchmark datsets
    elif dataset_name == "peptides-func":
        datasets = [LRGBDataset(root=dir, name='Peptides-func', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif dataset_name == "peptides-struct":
        datasets = [LRGBDataset(root=dir, name='Peptides-struct', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    # elif dataset_name == "pascalvoc-sp":
    #     datasets = [LRGBDataset(root=dir, name='PascalVOC-SP', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    # elif dataset_name == "coco-sp":
    #     datasets = [LRGBDataset(root=dir, name='COCO-SP', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    # elif dataset_name == "pcqm-contact":
    #     datasets = [LRGBDataset(root=dir, name='PCQM-Contact', split=split, pre_transform=transform) for split in ["train", "val", "test"]]
        
    elif "qm9" in dataset_name:
        dataset = QM9(root=dir, pre_transform=transform)
        indices = load_indices("QM9", config)
        datasets = [dataset[indices["train"][0]], dataset[indices["val"][0]], dataset[indices["test"][0]]]
    else:
        raise NotImplementedError("Unknown dataset")
        
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_model(args, num_classes, num_vertex_features, num_tasks):
    node_feature_dims = []
    model = args.model.lower()
    dataset_name = args.dataset.lower()

    # Load node and edge encoder
    if dataset_name == "zinc"and not args.do_drop_feat:
        node_feature_dims.append(21)
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
        edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[4])
    elif dataset_name in ["peptides-struct", "peptides-func", "ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"] and not args.do_drop_feat:
        node_feature_dims += get_atom_feature_dims()
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim)
    elif "qm9" in dataset_name:
        node_feature_dims += [2, 2, 2, 2, 2, 10, 1, 1, 1, 1, 5]
        edge_feature_dims = [2, 2, 2, 1]
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
        edge_encoder =  EdgeEncoder(emb_dim=args.emb_dim, feature_dims=edge_feature_dims)
    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x
            
    # Load model
    if model in ["gin", "gcn", "gat"]:  
        return MPNN(num_classes, 
                    num_tasks, 
                    args.num_mp_layers, 
                    args.emb_dim, 
                    gnn_type = model, 
                    drop_ratio = args.drop_out, 
                    JK = "last", 
                    graph_pooling = args.pooling, 
                    edge_encoder=edge_encoder, 
                    node_encoder=node_encoder, 
                    num_mlp_layers = args.num_mlp_layers, 
                    residual=args.use_residual, 
                    activation=args.activation)
    elif args.model.lower() == "mlp":
            return MLP(num_node_level_layers = args.num_n_layers,
                       num_graph_level_layers = args.num_g_layers,
                       node_encoder = node_encoder, 
                       emb_dim = args.emb_dim, 
                       num_classes = num_classes, 
                       num_tasks = num_tasks, 
                       dropout_rate = args.drop_out, 
                       graph_pooling = args.pooling, 
                       activation = args.activation)
    else: 
        raise ValueError("Unknown model name")

    return model

def get_optimizer_scheduler(model, args):
    lr = args.lr
    scheduler_name = args.scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)

    match scheduler_name:
        case 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        args.scheduler_decay_steps,
                                                        gamma=args.scheduler_decay_rate)
        case 'None':
            scheduler = None
        case "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode='min',
                                                                    factor=args.scheduler_decay_rate,
                                                                    patience=args.scheduler_patience,
                                                                    verbose=True)
        case "Cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps = args.warmup_steps, 
                                                        num_training_steps = args.epochs)
        case _:
            raise NotImplementedError(f'Scheduler {scheduler_name} is not currently supported.')

    return optimizer, scheduler

def get_loss(dataset_name):
    metric_method = None
    dataset_name_lowercase = dataset_name.lower()
    if dataset_name_lowercase in ["zinc", "peptides-struct"] or "qm9" in dataset_name_lowercase:
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif dataset_name_lowercase in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(dataset_name)
    elif dataset_name_lowercase in ["csl"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif dataset_name_lowercase in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(dataset_name)
    elif dataset_name_lowercase == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(dataset_name)
    elif dataset_name_lowercase in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(dataset_name)
    elif dataset_name == "peptides-func":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap" 
    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method