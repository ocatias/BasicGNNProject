import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from Misc.utils import PredictionType

def get_tracking_dict():
    return {"correct_classifications": 0, "y_preds":[], "y_true":[],  "total_loss":0, "batch_losses":[]}

def compute_loss_predictions(batch, model, metric, device, loss_fn, tracking_dict, prediction_type):  
    if batch.edge_attr is not None and len(batch.edge_attr.shape) == 1:
        batch.edge_attr = batch.edge_attr.view(-1, 1)

    batch = batch.to(device)
    predictions = model(batch)
    
    if prediction_type == PredictionType.EDGE_PREDICTION:
        y = batch.edge_label
    else:
        y = batch.y
        
    nr_predictions =  y.shape[0]

    if model.num_tasks == 1:
        y = y.view(nr_predictions, -1)
    else:
        y = y.view(nr_predictions, model.num_tasks)

    if y.shape[1] == 1 and metric == "accuracy":
        y = F.one_hot(torch.squeeze(y, 1), 10)
        
    is_labeled = y == y

    if y.dtype == torch.int64:
        y = y.float()
    ground_truth = y

    if metric in ['accuracy']:
        loss = loss_fn(predictions, ground_truth)  
    else:
        if is_labeled.shape == predictions.shape:
            loss = loss_fn(predictions[is_labeled], ground_truth.float()[is_labeled])
        else:
            loss = loss_fn(predictions, ground_truth)
    if metric == 'accuracy':
        tracking_dict["correct_classifications"] += torch.sum(predictions.argmax(dim=1)==ground_truth.argmax(dim=1)).item()

    tracking_dict["y_preds"] += predictions.cpu()
    tracking_dict["y_true"] += ground_truth.cpu()
    tracking_dict["batch_losses"].append(loss.item())
    tracking_dict["total_loss"] += loss.item()
    return loss

def compute_final_tracking_dict(tracking_dict, output_dict, loader, metric, metric_method, train=False):
    output_dict["total_loss"] = tracking_dict["total_loss"] / len(loader.dataset)
    if train:
        return output_dict

    if metric == 'accuracy':
        output_dict["accuracy"] = tracking_dict["correct_classifications"] / len(loader.dataset)
    elif "(ogb)" in metric:
        y_preds = torch.stack(tracking_dict["y_preds"])
        y_true = torch.stack(tracking_dict["y_true"])

        if len(y_preds.shape) == 1:
            y_preds = torch.unsqueeze(y_preds, dim = 1)
            y_true = torch.unsqueeze(y_true, dim = 1)

        output_dict[metric] = metric_method(y_true, y_preds)[metric.replace(" (ogb)", "")]
        
    elif metric == 'mae':
        y_preds = torch.concat(tracking_dict["y_preds"])
        y_true = torch.concat(tracking_dict["y_true"])
        y_preds = torch.unsqueeze(y_preds, dim = 1)
        y_true = torch.unsqueeze(y_true, dim = 1)
        l1 = torch.nn.L1Loss()
        output_dict["mae"] = float(l1(y_preds, y_true))
    elif metric == 'ap':
        sigmoid = torch.nn.Sigmoid()
        y_preds = sigmoid(torch.stack(tracking_dict["y_preds"]))
        y_true = torch.stack(tracking_dict["y_true"])
        if len(y_preds.shape) == 1:
            y_preds = torch.unsqueeze(y_preds, dim = 1)
            y_true = torch.unsqueeze(y_true, dim = 1)
            
        # From https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py (MIT License)

        ap_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            
            true_values = sum(y_true[:,i] == 1)
            false_values = sum(y_true[:,i] == 0)
            
            assert true_values > 0 and false_values > 0
     
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            ap = average_precision_score(y_true[is_labeled,i], y_preds[is_labeled,i])

            ap_list.append(ap)
        ap =  sum(ap_list)/len(ap_list)
        output_dict["ap"] = float(ap)
    elif metric == 'f1':
        y_preds = torch.stack(tracking_dict["y_preds"])
        y_preds = torch.argmax(y_preds, 1)
        y_target = torch.concat(tracking_dict['y_true'])      
        f1 = metric_method(y_preds, y_target)
        output_dict["f1"] = float(f1)
    elif metric == "mrr":
        import numpy as np
        y_preds = torch.concat(tracking_dict["y_preds"])
        y_target = torch.concat(tracking_dict['y_true'])  
        index = np.arange(y_target.shape[0])
        y_pos_index = index[y_target == 1]
        y_neg_index = index[y_target != 1]
        y_pos = y_preds[y_pos_index]
        y_neg = y_preds[y_neg_index]
        mrr = metric_method(y_pos, y_neg)
        output_dict["mrr"] = float(mrr)
        
    else:
        raise Exception("Unknown metric name")
    
    return output_dict

def train(model, device, train_loader, optimizer, loss_fct, eval_name, tracker, metric_method, prediction_type):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """
    model.train()

    tracking_dict = get_tracking_dict()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss_predictions(batch, model, eval_name, device, loss_fct, tracking_dict, prediction_type)

        loss.backward()
        optimizer.step()

        if tracker is not None:
            tracker.log({"Train/BatchLoss": loss.item()})
            
    return compute_final_tracking_dict(tracking_dict, {}, train_loader, eval_name, metric_method=metric_method, train=True)

def eval(model, device, loader, loss_fn, eval_name, metric_method, prediction_type):
    """
        Evaluates a model over all the batches of a data loader.
    """
    model.eval()

    tracking_dict = get_tracking_dict()
    for batch in loader:
        with torch.no_grad():
            compute_loss_predictions(batch, model, eval_name, device, loss_fn, tracking_dict, prediction_type)

    eval_dict = compute_final_tracking_dict(tracking_dict, {}, loader, eval_name, metric_method=metric_method)
    
    return eval_dict

def step_scheduler(scheduler, scheduler_name, val_loss):
    """
        Steps the learning rate scheduler forward by one
    """
    if scheduler_name in ["StepLR", "Cosine"]:
        scheduler.step()
    elif scheduler_name == 'None':
        pass
    elif scheduler_name == "ReduceLROnPlateau":
         scheduler.step(val_loss)
    else:
        raise NotImplementedError(f'Scheduler {scheduler_name} is not currently supported.')
