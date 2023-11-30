import os
import csv
import random
from glob import escape

import torch
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, GNNBenchmarkDataset, TUDataset
import torch.optim as optim
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.utils.features import get_atom_feature_dims

from Misc.count_node_degree import AddNodeDegree
from Misc.count_triangles import CountTriangles
from Misc.dataloader import DataLoader
from Misc.dataset_pyg_custom import PygGraphPropPredDatasetCustom, FilterMaxGraphSize, ComposeFilters
from Misc.transform_to_k_wl import TransforToKWl
from Models.gnn import GNN
from Models.encoder import NodeEncoder, EdgeEncoder, ZincAtomEncoder, EgoEncoder
from Models.mlp import MLP
from Misc.drop_features import DropFeatures
from Misc.add_zero_edge_attr import AddZeroEdgeAttr, AddZeroNodeAttr, DebugTransform
from Misc.pad_node_attr import PadNodeAttr


def get_filters(args):
    filters = []
    if args.filter_data_max_graph_size > 0:
        filters.append(FilterMaxGraphSize(args.filter_data_max_graph_size))
    return ComposeFilters(filters)


def get_transform(args, split=None):
    transforms = []

    if args.do_drop_feat:
        emb_dim = args.emb_dim
        if args.transform_k_wl:
            emb_dim -= 1
        transforms.append(DropFeatures(emb_dim))
    if args.dataset.lower() == "csl":
        transforms.append(OneHotDegree(5))
    if args.transform_k_wl:
        if args.sequential_k_wl and int(args.transform_k_wl) == 3:
            transforms.append(TransforToKWl(k=2,
                                            turbo=args.k_wl_turbo,
                                            max_group_size=args.k_wl_turbo_max_group_size,
                                            agg_function_features=args.k_wl_pool_function,
                                            set_based=bool(args.k_wl_set_based),
                                            modify=not bool(args.sequential_k_wl)))
        transforms.append(TransforToKWl(k=args.transform_k_wl,
                                        turbo=args.k_wl_turbo,
                                        max_group_size=args.k_wl_turbo_max_group_size,
                                        agg_function_features=args.k_wl_pool_function,
                                        set_based=bool(args.k_wl_set_based),
                                        modify=not bool(args.sequential_k_wl)))
        if args.sequential_k_wl:
            if args.add_node_degree:
                transforms.append(AddNodeDegree())
            else:
                transforms.append(AddZeroNodeAttr(1))

            transforms.append(AddZeroEdgeAttr(1))
    # Pad features if necessary (needs to be done after adding additional features from other transformation)
    if args.add_num_triangles:
        transforms.append(CountTriangles())
    if args.dataset.lower() in ["csl", "ptc_mr", "ptc_fm", 'mutag', 'imdb-binary', 'imdb-multi',
                                'enzymes'] and not args.transform_k_wl:
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))

    return Compose(transforms)


def escape_dir(s):
    return s.replace("\n", "").replace(' ', '').replace('=', '-').replace(',', '-').replace('[', '').replace(']', '')


def load_dataset(args, config):
    transform = get_transform(args)
    filter = get_filters(args)
    print(repr(transform))
    print(repr(filter))
    trafo_str = escape_dir(repr(transform))
    filter_str = escape_dir(repr(filter))
    dir = os.path.join(config.DATA_PATH, args.dataset, filter_str, trafo_str)

    if args.dataset.lower() == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in
                    ["train", "val", "test"]]
    elif args.dataset.lower() == "cifar10":
        datasets = [GNNBenchmarkDataset(name="CIFAR10", root=dir, split=split,
                                        pre_transform=Compose([ToUndirected(), transform])) for split in
                    ["train", "val", "test"]]
    elif args.dataset.lower() == "cluster":
        datasets = [GNNBenchmarkDataset(name="CLUSTER", root=dir, split=split, pre_transform=transform) for split in
                    ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21",
                                  "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv",
                                  "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDatasetCustom(root=dir, name=args.dataset.lower(), pre_transform=transform,
                                                pre_filters=filter)
        # memory_intense_pre_transform=True)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    elif args.dataset.lower() == "csl":
        # TODO try to get more than 0.1 acc on this
        all_idx = {}
        for section in ['train', 'val', 'test']:
            with open(os.path.join(config.SPLITS_PATH, "CSL", f"{section}.index"), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        dataset = GNNBenchmarkDataset(name="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[all_idx["train"][args.split]], dataset[all_idx["val"][args.split]],
                    dataset[all_idx["test"][args.split]]]
    elif args.dataset.lower() in ["exp", "cexp"]:
        dataset = PlanarSATPairsDataset(name=args.dataset, root=dir, pre_transform=transform)
        split_dict = dataset.separate_data(args.seed, args.split)
        datasets = [split_dict["train"], split_dict["valid"], split_dict["test"]]
    elif args.dataset.lower() in ["ptc_mr", "ptc_fm", 'mutag', 'imdb-binary', 'imdb-multi', 'enzymes']:
        print('dir', dir)
        dataset = TUDataset(root=escape(dir.replace('\\', '/')), name=args.dataset, pre_transform=transform,
                            pre_filter=filter, use_node_attr=True, use_edge_attr=True)

        split_idx = {'train': [], 'valid': [], 'test': []}
        random.seed(42)
        for i in range(len(dataset)):
            x = dataset.get(i)
            if random.random() < 0.5:
                split_idx['train'].append(i)
            elif random.random() < 0.5:
                split_idx['valid'].append(i)
            else:
                split_idx['test'].append(i)

        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    else:
        raise NotImplementedError("Unknown dataset")

    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)

    del transform
    return train_loader, val_loader, test_loader


def get_model(args, num_classes, num_vertex_features, num_tasks, uses_k_wl_transform, k_wl_separate_embedding):
    node_feature_dims = []

    model = args.model.lower()
    if args.dataset.lower() == "zinc" and not args.do_drop_feat:
        node_feature_dims.append(21)
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims,
                                   uses_k_wl_transform=uses_k_wl_transform,
                                   k_wl_separate=k_wl_separate_embedding)
        edge_encoder = EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[4], uses_k_wl_transform=uses_k_wl_transform,
                                   k_wl_separate=k_wl_separate_embedding)
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace",
                                  "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast",
                                  "ogbg-molfreesolv", "ogbg-mollipo"] and not args.do_drop_feat:

        node_feature_dims += get_atom_feature_dims()
        print("node_feature_dims: ", node_feature_dims)
        node_encoder, edge_encoder = \
            NodeEncoder(args.emb_dim, feature_dims=node_feature_dims,
                        uses_k_wl_transform=uses_k_wl_transform,
                        k_wl_separate=k_wl_separate_embedding), \
                EdgeEncoder(args.emb_dim,
                            uses_k_wl_transform=uses_k_wl_transform,
                            k_wl_separate=k_wl_separate_embedding)
    elif args.dataset.lower() in ["csl", "ptc_mr", "ptc_fm", 'mutag', 'imdb-binary', 'imdb-multi', 'enzymes']:
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=[300, 100, 100, 100, 100, 100, 100],
                                   uses_k_wl_transform=uses_k_wl_transform,
                                   k_wl_separate=k_wl_separate_embedding)
        edge_encoder = EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[100, 100, 100],
                                   uses_k_wl_transform=uses_k_wl_transform,
                                   k_wl_separate=k_wl_separate_embedding)

    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x

    if model in ["gin", "gcn", "gat"]:
        return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim,
                   gnn_type=model, virtual_node=args.use_virtual_node, drop_ratio=args.drop_out, JK="last",
                   graph_pooling=args.pooling, edge_encoder=edge_encoder, node_encoder=node_encoder,
                   use_node_encoder=args.use_node_encoder, num_mlp_layers=args.num_mlp_layers,
                   sequential_k_wl=args.sequential_k_wl, k_wl=args.transform_k_wl)
    elif args.model.lower() == "mlp":
        return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim,
                   num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling)
    else:  # Probably don't need other models
        raise ValueError("Unknown model name")

    return model


def get_optimizer_scheduler(model, args, finetune=False):
    if finetune:
        lr = args.lr2
    else:
        lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=args.lr_scheduler_decay_rate,
                                                               patience=args.lr_schedule_patience,
                                                               verbose=True)
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return optimizer, scheduler


def get_loss(args):
    metric_method = None
    if args.dataset.lower() == "zinc":
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif args.dataset.lower() in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() in ["cifar10", "csl", "exp", "cexp"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif args.dataset.lower() in ["ptc_mr", "ptc_fm", 'mutag', 'imdb-binary',
                                  'imdb-multi', 'enzymes']:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy"
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox"
                                                                                                  "ogbg-molsider",
                                  "ogbg-moltoxcast", ]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)"
        metric_method = get_evaluator(args.dataset)
    else:
        raise NotImplementedError("No loss for this dataset")

    return {"loss": loss, "metric": metric, "metric_method": metric_method}


def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method
