"""
Helper functions that do argument parsing for experiments.
"""

import argparse
import yaml
import sys
from copy import deepcopy

from Misc.config import config
from Misc.utils import transform_dict_to_args_list


def add_general_arguments(parser):
    """
    Arguments that are always necessary for training a model
    """
    parser.add_argument('--config', dest='config_file', type=argparse.FileType(mode='r'),
                    help='Path to a config file that should be used for this experiment. '
                    + 'CANNOT be combined with explicit arguments')
    parser.add_argument('--model', type=str, default='GIN',
                    help='Model to use (default: GIN; other options: GCN, MLP)')
    parser.add_argument('--dataset', type=str, default="ZINC",
                    help='Dataset name (default: ZINC; other options: CSL and most datasets from ogb, see ogb documentation)')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau',
                    help='Learning rate decay scheduler (default: ReduceLROnPlateau; other options: StepLR, None; For details see PyTorch documentation)')
    parser.add_argument('--tracking', type=int, default=config.use_wandb_tracking,
                        help=f'If 0 runs without tracking (Default: {str(config.use_wandb_tracking)})')
    
def add_training_arguments(parser):
    parser.add_argument('--max_time', type=float, default=12,
                        help='Max time (in hours) for one run')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--device', type=int, default=0,
                    help='Which gpu to use if any (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Input batch size for training (default: 32)')

def add_data_arguments(parser):
    parser.add_argument('--drop_feat', type=int, default=0,
                        help="Set to 1 to drop all edge and vertex features from the graph (default: 0)")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--split', type=int, default=0,
                        help='Split for cross validation (default: 0)')
    
def add_model_arguments(parser, model):
    parser.add_argument('--drop_out', type=float, default=0.0,
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Dimensionality of hidden units in models (default: 64)')
    parser.add_argument('--pooling', type=str, default="mean",
                        help='Graph pooling operation to use (default: mean; other options: sum)')
    parser.add_argument('--activation', type=str, default="relu",
                        help='Activation function (default: relu; other options: elu, id, sigmoid, tanh)')
    
    if model in ["GCN", "GIN", "GAT"]:
        parser.add_argument('--num_mp_layers', type=int, default=5,
                        help='Number of message passing layers (default: 5) ')
        parser.add_argument('--num_mlp_layers', type=int, default=2,
                            help='Number of layers in the MLP that performs predictions on the embedding computed by the GNN (default: 1)')
        # parser.add_argument('--virtual_node', type=int, default=0,
        #                     help='Set 1 to use a virtual node, that is a node that is adjacent to every node in the graph (default: 0)')
        parser.add_argument('--residual', type=int, default=0,
                            help='Set 1 for a residual connection in MPNNs (default: 0)')
        
    elif model == "MLP":
        parser.add_argument('--num_n_layers', type=int, default=5,
                        help='Number of fully connected layers that are applied to each node (default: 5)')
        parser.add_argument('--num_g_layers', type=int, default=5,
                        help='Number of fully connected layers that are applied to the whole graph (default: 5)')
    else:
        raise NotImplementedError

def parse_with_config_and_parsed_args(parser, passed_args, do_parse_only_known):
    # Load partial args instead of command line args (if they are given)
    
    # [Case parser is given just enough args to infer the missing args]: ignore the unknown commandline arguments
    if do_parse_only_known:
        if passed_args is None:
            args, _ = parser.parse_known_args()
        else:
            args, _ = parser.parse_known_args(transform_dict_to_args_list(passed_args))  
            
    # [Case parser has infered the missing args]: parse every command line argument to ensure that all arguments make sense
    else:      
        if passed_args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(transform_dict_to_args_list(passed_args)) 

    # https://codereview.stackexchange.com/a/79015
    # If a config file is provided, write it's values into the arguments
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
                arg_dict[key] = value
    return args

def add_scheduler_args(parser, scheduler):
    if scheduler in ["ReduceLROnPlateau", "StepLR"]:
        parser.add_argument('--scheduler_decay_rate', type=float, default=0.5,
                            help='(For LR schedulers that are not None) Strength of lr decay (default: 0.5)')

    if scheduler == "StepLR":
        parser.add_argument('--scheduler_decay_steps', type=int, default=50,
                            help='(For StepLR scheduler) number of epochs between lr decay (default: 50)')
    elif scheduler == "ReduceLROnPlateau":
        parser.add_argument('--scheduler_min_lr', type=float, default=1e-5,
                            help='(For ReduceLROnPlateau scheduler) mininum learnin rate (default: 1e-5)')
        parser.add_argument('--scheduler_patience', type=int, default=10,
                            help='(For ReduceLROnPlateau scheduler) number of epochs without improvement until the LR will be reduced')

args_with_use = ["tracking", "residual"]
args_with_do = ["drop_feat"]
def int_to_bool_args(args):
    """
    Some commandline arguments are booleans that are set as 0 (false) or 1 (true)
    This function takes these arguments, renames them (by adding use or do as prefix) and transforms them into booleans
    For example:
    "residual" (0 or 1) -> "use_residual" (False or True) 
    """
    for arg in args_with_use:
        if arg in args.__dict__:
            args.__dict__["use_" + arg] = args.__dict__[arg] == 1
    for arg in args_with_do:
        if arg in args.__dict__:
            args.__dict__["do_" + arg] = args.__dict__[arg] == 1
    

def parse_args(passed_args=None):
    """
    Parse command line arguments. Allows either a config file (via "--config path/to/config.yaml")
    or for all parameters to be set directly.
    A combination of these is NOT allowed.
    Partially from: https://github.com/twitter-research/cwn/blob/main/exp/parser.py
    """

    # Get general arguments before we try to parse the specifics
    initial_parser = argparse.ArgumentParser(description='An experiment.')
    add_general_arguments(initial_parser)
    initial_args = parse_with_config_and_parsed_args(initial_parser, passed_args, do_parse_only_known=True)
    
    parser = argparse.ArgumentParser(description='An experiment.')
    add_general_arguments(parser)
    add_scheduler_args(parser, initial_args.scheduler)
    add_training_arguments(parser)
    add_model_arguments(parser, initial_args.model)
    add_data_arguments(parser)
    args = parse_with_config_and_parsed_args(parser, passed_args, do_parse_only_known=False)
    
    int_to_bool_args(args)
    return args
