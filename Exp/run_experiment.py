"""
Runs an experiment: searches for hyperparameters and then trains the final model multiple times
"""
import argparse
import gc
import os
import glob
import json
import copy
import sys
from collections import defaultdict
import random

import torch
import yaml
from sklearn.model_selection import ParameterGrid
import numpy as np

from Exp.run_model import run
from Misc.config import config
from Misc.utils import transform_dict_to_args_list

keys_to_avg = ["runtime_hours", "parameters", "val", "test"]

# How often an exception can be thrown by training / evaluation without the experiment stopping
allowed_nr_errors = 5000

binary_class_ogb_datasets = ["molbace", "molbbbp", "molclintox", "molmuv", "molpcba", "molsider", "moltox21",
                             "moltoxcast", "molhiv", "molchembl"]
binary_class_datsets = binary_class_ogb_datasets
regression_ogb_datasets = ["molesol", "molfreesolv", "mollipo"]
regression_datsets = regression_ogb_datasets + ["zinc"]
ogb_datasets = binary_class_ogb_datasets + regression_ogb_datasets


def get_directory(args):
    """
    :returns: Directory that should store the experiment results
    """
    return os.path.join(config.RESULTS_PATH, f"{args.dataset}_{os.path.split(args.grid_file)[-1]}")


def get_paths(args, split):
    """
    :returns: all relevant paths in which experiment results will be stored
    """
    directory = get_directory(args)

    if args.folds > 1:
        results_path = os.path.join(directory, f"split_{split}")
    else:
        results_path = directory
    hyperparams_path = os.path.join(results_path, "Hyperparameters")
    final_eval_path = os.path.join(results_path, "FinalResults")
    errors_path = os.path.join(results_path, "Errors")
    return directory, results_path, hyperparams_path, final_eval_path, errors_path


def dict_to_args(args_dict):
    """
    Transform dict to list of args
    :returns: list of args
    """
    list_args = []
    for key, value in args_dict.items():
        # The case with "" happens if we want to pass an argument that has no parameter
        if value != "":
            list_args += [key, str(value)]
        else:
            list_args += [key]

    print(list_args)
    return list_args


def store_error(error, errors_path, file_name, additional_info={}):
    """
    Stores the error dictionary
    :returns: number of errors found in total
    """
    error_dict = {
        "error_type": str(type(error)),
        "error": str(error)}
    error_dict = error_dict | additional_info
    nr_prev_errors = len(glob.glob(os.path.join(errors_path, '*')))
    output_path = os.path.join(errors_path, f"error_{nr_prev_errors}_{file_name}.json")
    with open(output_path, "w") as file:
        json.dump(error_dict, file, indent=4)
    return nr_prev_errors + 1


def parse_args(passed_args):
    parser = argparse.ArgumentParser(description='An experiment.')
    parser.add_argument('-grid', dest='grid_file', type=str,
                        help="Path to a .yaml file that contains the parameters grid.")
    parser.add_argument('-dataset', type=str)
    parser.add_argument('--candidates', type=int, default=20,
                        help="Number of parameter combinations to try per fold.")
    parser.add_argument('--repeats', type=int, default=10,
                        help="Number of times to repeat the final model training")
    parser.add_argument('--folds', type=int, default="1",
                        help='Number of folds, setting this to something other than 1, means we will treat this as cross validation')
    parser.add_argument('--device', type=int, default=-1,
                        help='Overwrites the setting for device from grid')
    parser.add_argument('--transform_k_wl', type=int, default=-1,
                        help='Overwrites the setting for transform_k_wl from grid')

    if passed_args is None:
        return parser.parse_args(passed_args)
    else:
        print(transform_dict_to_args_list(passed_args))
        return parser.parse_args(transform_dict_to_args_list(passed_args))


def create_directories(directory, args):
    if os.path.isdir(directory):
        return

    os.mkdir(directory)
    for split in range(args.folds):
        directory, results_path, hyperparams_path, final_eval_path, errors_path = get_paths(args, split)

        if not os.path.isdir(results_path):
            os.mkdir(results_path)
        os.mkdir(hyperparams_path)
        os.mkdir(final_eval_path)
        os.mkdir(errors_path)


def select_param(parameter_space, hyperparams_path):
    """
    Select new parameters from the parameter space to evaluate
    """
    param = None
    already_checked_params = False
    stored_params = []

    for param_file in glob.glob(os.path.join(hyperparams_path, "*.json")):
        with open(param_file) as file:
            stored_params.append(yaml.safe_load(file))

    while param is None:
        param = parameter_space[0]
        del parameter_space[0]
        already_checked_params = False

        # Check if we have tested that param before
        for curr_param_to_check in stored_params:
            same_param = True
            for key, value in param.items():

                print(f"--{str(key)}", curr_param_to_check["params"][f"--{str(key)}"], str(value))
                if str(curr_param_to_check["params"][f"--{str(key)}"]) != str(value):
                    same_param = False
                    break

            if same_param:
                print(f"Already tried {param} and got validation score of {curr_param_to_check['val']}")
                param = None
                break

    return param


def create_param_dict(param, args, split):
    param_dict = {
        "--dataset": args.dataset
    }

    for key, value in param.items():
        param_dict["--" + str(key)] = str(value)

    if args.folds > 1:
        param_dict["--split"] = split

    return param_dict


def evaluate_params(param_dict, hyperparams_path):
    """
    Trains a model for the given hyperparameters and stores the results
    :returns: evaluation mode
    """
    print('training on ', param_dict)
    result_dict = run(param_dict)
    output_path = os.path.join(hyperparams_path, f"params_{len(glob.glob(os.path.join(hyperparams_path, '*')))}.json")
    storage_dict = {"params": param_dict}
    storage_dict.update(copy.deepcopy(result_dict))

    with open(output_path, "w") as file:
        json.dump(storage_dict, file, indent=4)
    del result_dict
    del storage_dict
    gc.collect()
    torch.cuda.empty_cache()
    # return result_dict["mode"]


def load_best_params(mode, hyperparams_path):
    """
    Loads the best params from disk
    :returns: the best params
    """
    param_files = glob.glob(os.path.join(hyperparams_path, "*.json"))
    # print(hyperparams_path)
    if len(param_files) == 0:
        raise Exception("No hyperparameters found.")

    best_result_val = None
    for param_file in param_files:
        with open(param_file) as file:
            dict = yaml.safe_load(file)

            if mode is None:
                mode = dict["mode"]

            if best_result_val is None or (mode == "min" and dict["val"] < best_result_val) or (
                    mode == "max" and dict["val"] > best_result_val):
                best_params = dict["params"]
                best_result_val = dict["val"]

    print(f"Best params have score {best_result_val:.4f} and are:\n{best_params}")
    return best_params


def run_final_evaluation(args, final_eval_path, best_params):
    """
    Train and evaluate a model with the best parameters multiple times
    """
    prev_evaluation = glob.glob(os.path.join(final_eval_path, "*.json"))
    nr_tries = args.repeats - len(prev_evaluation)

    for iteration in range(nr_tries):
        print(f"Evaluation: {iteration + len(prev_evaluation) + 1} / {nr_tries}")
        best_params["--seed"] = iteration
        result_dict = run(best_params)

        output_dict = {}
        output_dict["params"] = best_params
        output_dict.update(copy.deepcopy(result_dict))

        output_path = os.path.join(final_eval_path, f"eval_{len(glob.glob(os.path.join(final_eval_path, '*')))}.json")
        # print(output_path)
        with open(output_path, "w") as file:
            json.dump(output_dict, file, indent=4)

        del output_dict
        del result_dict
        gc.collect()


def collect_eval_results(args, mode):
    """
    Collects final results (if crossvalidation: over all folds)
    :returns: dictionary of the tracked metrics
    """

    final_results = defaultdict(lambda: [])

    for split in range(args.folds):
        directory, results_path, hyperparams_path, final_eval_path, errors_path = get_paths(args, split)
        eval_paths = list(glob.glob(os.path.join(final_eval_path, '*.json')))

        # Collect results
        for eval_path in eval_paths:
            with open(eval_path) as file:
                result_dict = json.load(file)

                # If we have not run any models yet we need to load it
                if mode is None:
                    mode = result_dict["mode"]

            for key in result_dict.keys():
                final_results[key].append(result_dict[key])
    return final_results


def compute_store_output(directory, final_results):
    """
    Computes average and standard deviation of tracked metrics and stores them
    :returns: dictionary of avg, std of metrics
    """
    output = {}

    for key in keys_to_avg:
        avg, std = np.average(final_results[key]), np.std(final_results[key])
        output[f"{key}-avg"] = avg
        output[f"{key}-std"] = std
        print(f"{key}:\t{avg:.4f}±{std:4f}")

    output_path = os.path.join(directory, "final.json")

    with open(output_path, "w") as file:
        json.dump(output, file, indent=4)

    return output


def find_eval_params(args, grid, split):
    """
    Finds the best hyperparameters and performs the final validation for a single split of the data.S
    Results are stored on disk.
    :returns: evaluation mode
    """
    parameter_space = copy.deepcopy(grid)
    random.shuffle(parameter_space)
    # print('parameter_space', parameter_space)
    print(f"Grid contains {len(grid)} combinations")
    _, _, hyperparams_path, final_eval_path, errors_path = get_paths(args, split)

    print("Searching for hyperparameters")
    mode = None

    # In case we have restarted the script: check how many parameter configurations we have already tried
    previously_tested_params = glob.glob(os.path.join(hyperparams_path, "*.json"))
    nr_tries = min(args.candidates, len(grid)) - len(previously_tested_params)
    print(nr_tries)
    for c in range(nr_tries):
        # Evaluate a parmeter configuration

        np.random.seed()
        if len(parameter_space) == 0:
            break

        # Select the parameter configuration
        param = select_param(parameter_space, hyperparams_path)
        param_dict = create_param_dict(param, args, split)

        try:
            if len(grid) == 1 or args.candidates == 1:
                print('breaking because no paratemeres left')
                # Don't search for params if there is only one candidate
                break

            # Evaluate.
            evaluate_params(param_dict, hyperparams_path)

        except Exception as e:
            nr_prev_errors = store_error(e, errors_path, "hyperparam_search", {"params": param_dict})
            print(e.__str__())
            if nr_prev_errors > allowed_nr_errors:
                raise Exception("Too many training runs crashed.")

    print("Finished search.\n", "Selecting best parameters.")

    # if len(grid) == 1 or args.candidates == 1:
    #     # Don't load parameters if we have not searched for them
    #     best_params = param_dict
    #     print(f"Only one parameter pair / candidate (not searching for params)")
    #
    # else:
    #     best_params = load_best_params(mode, hyperparams_path)
    #
    # print("Evaluating the best parameters.")
    # run_final_evaluation(args, final_eval_path, best_params)
    # return mode


def main(passed_args=None):
    args = parse_args(passed_args)

    with open(args.grid_file, 'r') as file:
        grid_raw = yaml.safe_load(file)
    if args.device != -1:
        grid_raw['device'] = [args.device]
    if args.transform_k_wl != -1:
        grid_raw['transform_k_wl'] = [args.transform_k_wl]

    grid = list(ParameterGrid(grid_raw))
    directory = get_directory(args)
    create_directories(directory, args)

    try:
        for split in range(args.folds):
            find_eval_params(args, grid, split)

        # print("Collect evaluation results.")
        # final_results = collect_eval_results(args, mode)
        # output = compute_store_output(directory, final_results)
        # print(output)

    except Exception as e:
        # Detected an error: store the params and the error
        _, _, _, _, errors_path = get_paths(args, split)
        nr_prev_errors = store_error(e, errors_path, "run_exp", {"args": str(args)})

        print("An error has been thrown in the run_experiment main method. Exiting.")
        raise Exception


if __name__ == "__main__":
    main()
