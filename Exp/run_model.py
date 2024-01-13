"""
Trains and evaluates a model a single time for given hyperparameters.
"""

import random
import time
import os
from uuid import uuid4

import wandb
import torch
import numpy as np

from Exp.parser import parse_args
from Misc.config import config
from Misc.utils import list_of_dictionary_to_dictionary_of_lists
from Exp.preparation import load_dataset, get_model, get_optimizer_scheduler, get_loss
from Exp.training_loop_functions import train, eval, step_scheduler


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def track_epoch(epoch, metric_name, train_result, val_result, test_result, lr):
    wandb.log({
        "Epoch": epoch,
        "Train/Loss": train_result["total_loss"],
        "Val/Loss": val_result["total_loss"],
        f"Val/{metric_name}": val_result[metric_name],
        "Test/Loss": test_result["total_loss"],
        f"Test/{metric_name}": test_result[metric_name],
        "LearningRate": lr
    })


def print_progress(train_loss, val_loss, test_loss, metric_name, val_metric, test_metric):
    print(f"\tTRAIN\t loss: {train_loss:6.4f}")
    print(f"\tVAL\t loss: {val_loss:6.4f}\t  {metric_name}: {val_metric:10.4f}")
    print(f"\tTEST\t loss: {test_loss:6.4f}\t  {metric_name}: {test_metric:10.4f}")


def main(args, cross_val_i):
    print(args)
    device = args.device
    os.environ['DEVICE'] = str(device)
    if device.isdecimal():
        device = int(device)
    use_tracking = args.use_tracking

    set_seed(args.seed)
    train_loader, val_loader, test_loader = load_dataset(args, config, cross_val_i)
    num_classes, num_vertex_features = train_loader.dataset.num_classes, train_loader.dataset.num_node_features

    if args.dataset.lower() == "zinc" or "ogb" in args.dataset.lower():
        num_classes = 1

    try:
        num_tasks = train_loader.dataset.num_tasks
    except:
        num_tasks = 1

    print(f"#Features: {num_vertex_features}")
    print(f"#Classes: {num_classes}")
    print(f"#Tasks: {num_tasks}")

    model = get_model(args, num_classes, num_vertex_features, num_tasks, args.transform_k_wl > 0,
                      bool(args.k_wl_separate_embedding))
    nr_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(model, args)
    loss_dict = get_loss(args)
    loss_fct = loss_dict["loss"]
    eval_name = loss_dict["metric"]
    metric_method = loss_dict["metric_method"]

    if use_tracking:
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            config=args,
            project=config.project)

    print("Begin training.\n")
    time_start = time.time()
    train_results, val_results, test_results = [], [], []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")
        train_result = train(model, device, train_loader, optimizer, loss_fct, eval_name, use_tracking,
                             metric_method=metric_method)
        val_result = eval(model, device, val_loader, loss_fct, eval_name, metric_method=metric_method)
        test_result = eval(model, device, test_loader, loss_fct, eval_name, metric_method=metric_method)

        train_results.append(train_result)
        val_results.append(val_result)
        test_results.append(test_result)
        print(test_result)
        print_progress(train_result['total_loss'], val_result['total_loss'], test_result['total_loss'], eval_name,
                       val_result[eval_name], test_result[eval_name])

        if use_tracking:
            track_epoch(epoch, eval_name, train_result, val_result, test_result, optimizer.param_groups[0]['lr'])

        step_scheduler(scheduler, args, val_result["total_loss"])

        # EXIT CONDITIONS
        if optimizer.param_groups[0]['lr'] < args.min_lr:
            print("\nLR reached minimum: exiting.")
            break

        runtime = (time.time() - time_start) / 3600
        if args.max_time > 0 and runtime > args.max_time:
            print("\nMaximum training time reached: exiting.")
            break

    # Final result
    train_results = list_of_dictionary_to_dictionary_of_lists(train_results)
    val_results = list_of_dictionary_to_dictionary_of_lists(val_results)
    test_result = list_of_dictionary_to_dictionary_of_lists(test_results)

    if eval_name in ["mae", "rmse (ogb)"]:
        best_val_epoch = np.argmin(val_results[eval_name])
        best_test_epoch = np.argmin(test_result[eval_name])
        mode = "min"
    else:
        best_val_epoch = np.argmax(val_results[eval_name])
        best_test_epoch = np.argmax(test_result[eval_name])
        mode = "max"

    loss_train, loss_val, loss_test = train_results['total_loss'][best_val_epoch], val_results['total_loss'][
        best_val_epoch], test_result['total_loss'][best_val_epoch]
    result_val, result_test = val_results[eval_name][best_val_epoch], test_result[eval_name][best_val_epoch]

    print("\n\nFinal Result:")
    print(f"\tRuntime: {runtime:.2f}h")
    print(f"\tBest epoch {best_val_epoch} / {args.epochs}")
    print_progress(loss_train, loss_val, loss_test, eval_name, result_val, result_test)

    if use_tracking:
        wandb.log({
            "Final/Train/Loss": loss_train,
            "Final/Val/Loss": loss_val,
            f"Final/Val/{eval_name}": result_val,
            "Final/Test/Loss": loss_test,
            f"Final/Test/{eval_name}": result_test,
            "BestTest/Val/Loss": val_results['total_loss'][best_test_epoch],
            f"BestTest/Val/{eval_name}": val_results[eval_name][best_test_epoch],
            "BestTest/Test/Loss": val_results['total_loss'][best_test_epoch],
            f"BestTest/Test/{eval_name}": test_result[eval_name][best_test_epoch],

        })

        wandb.finish()

    return {
        "mode": mode,
        "loss_train": float(loss_train),
        "loss_val": float(loss_val),
        "loss_test": float(loss_test),
        "val": float(result_val),
        "test": float(result_test),
        "runtime_hours": float(runtime),
        "epochs": int(epoch),
        "best_val_epoch": int(best_val_epoch),
        "parameters": int(nr_parameters),
        "details_train": train_results,
        "details_val": val_results,
        "details_test": test_results,
    }


def run(passed_args=None):
    args = parse_args(passed_args)
    if args.cross_validation > 0:
        cross_validation = range(args.cross_validation)
        cross_val_id = uuid4()
        args.cross_val_id = str(cross_val_id)
    else:
        cross_validation = [1]
    for i in cross_validation:
        args.cross_val_run = i
        torch.cuda.empty_cache()
        print(f'run {i}')
        print('memory stats', torch.cuda.memory_stats())
        main(args, i)


if __name__ == "__main__":
    run()
