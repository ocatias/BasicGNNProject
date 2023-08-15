"""
Helper functions that do argument parsing for experiments.
"""

import argparse
import yaml
import sys
from copy import deepcopy

from Misc.config import config
from Misc.utils import transform_dict_to_args_list

def parse_args(passed_args=None):
    """
    Parse command line arguments. Allows either a config file (via "--config path/to/config.yaml")
    or for all parameters to be set directly.
    A combination of these is NOT allowed.
    Partially from: https://github.com/twitter-research/cwn/blob/main/exp/parser.py
    """

    parser = argparse.ArgumentParser(description='An experiment.')

    # Config file to load
    parser.add_argument('--config', dest='config_file', type=argparse.FileType(mode='r'),
                        help='Path to a config file that should be used for this experiment. '
                        + 'CANNOT be combined with explicit arguments')

    parser.add_argument('--tracking', type=int, default=config.use_wandb_tracking,
                        help=f'If 0 runs without tracking (Default: {str(config.use_wandb_tracking)})')


    # Parameters to be set directly
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--split', type=int, default=0,
                        help='Split for cross validation (default: 0)')
    parser.add_argument('--dataset', type=str, default="ZINC",
                            help='Dataset name (default: ZINC; other options: CSL and most datasets from ogb, see ogb documentation)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    
    parser.add_argument('--device', type=str, default="0",
                    help='Which gpu to use if any (default: 0)')
    parser.add_argument('--model', type=str, default='GIN',
                    help='Model to use (default: GIN; other options: GCN, MLP)')
                    
    # LR SCHEDULER
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau',
                    help='Learning rate decay scheduler (default: ReduceLROnPlateau; other options: StepLR, None; For details see PyTorch documentation)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='Strength of lr decay (default: 0.5)')

    # For StepLR
    parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
                        help='(For StepLR scheduler) number of epochs between lr decay (default: 50)')

    # For ReduceLROnPlateau
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='(For ReduceLROnPlateau scheduler) mininum learnin rate (default: 1e-5)')
    parser.add_argument('--lr_schedule_patience', type=int, default=10,
                        help='(For ReduceLROnPlateau scheduler) number of epochs without improvement until the LR will be reduced')

    parser.add_argument('--max_time', type=float, default=12,
                        help='Max time (in hours) for one run')

    parser.add_argument('--drop_out', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Dimensionality of hidden units in models (default: 64)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of message passing layers (default: 5) or number of layers of the MLP')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='Number of layers in the MLP that performs predictions on the embedding computed by the GNN (default: 1)')
    parser.add_argument('--virtual_node', type=int, default=0,
                        help='Set 1 to use a virtual node, that is a node that is adjacent to every node in the graph (default: 0)')

    parser.add_argument('--pooling', type=str, default="mean",
                        help='Graph pooling operation to use (default: mean; other options: sum)')
    parser.add_argument('--node_encoder', type=int, default=1,
                        help="Set to 0 to disable to node encoder (default: 1)")

    parser.add_argument('--drop_feat', type=int, default=0,
                        help="Set to 1 to drop all edge and vertex features from the graph (default: 0)")

    parser.add_argument('--transform_k_wl', type=int, default=0,
                        help="Transform dataset to k-WL dataset. Specify K here. K must be 2 or 3")
    parser.add_argument('--k_wl_turbo', type=int, default=0,
                        help="Whether to use the turbo version of k-WL")
    parser.add_argument('--k_wl_turbo_max_group_size', type=int, default=-1,
                        help="Maximum size of a group considered by k-WL turbo. For efficiency. "
                             "Large graphs may be too large to compute.")
    parser.add_argument('--filter_data_max_graph_size', type=int, default=0,
                        help="What is the maximum size of graph to be used in training. 0 means no filter.")

    # Load partial args instead of command line args (if they are given)
    if passed_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(transform_dict_to_args_list(passed_args))        

    args.__dict__["use_tracking"] = args.tracking == 1
    args.__dict__["use_virtual_node"] = args.virtual_node == 1
    args.__dict__["use_node_encoder"] = args.node_encoder == 1
    args.__dict__["do_drop_feat"] = args.drop_feat == 1

    # https://codereview.stackexchange.com/a/79015
    # If a config file is provided, write it's values into the arguments
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
                arg_dict[key] = value

    return args
