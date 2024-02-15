import torch
import wandb
from ogb.graphproppred import Evaluator
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from Exp.preparation import get_evaluator

def get_tracking_dict():
    return {"correct_classifications": 0, "y_preds":[], "y_true":[],  "total_loss":0, "batch_losses":[]}

def compute_loss_predictions(batch, model, metric, device, loss_fn, tracking_dict):
    batch_size = batch.y.shape[0]

    # I think this was only necessary for ogbg-molpcba + CRE:
    # batch.edge_index = batch.edge_index.long()

    if batch.edge_attr is not None and len(batch.edge_attr.shape) == 1:
        batch.edge_attr = batch.edge_attr.view(-1, 1)

    batch = batch.to(device)
    predictions = model(batch)
    y = batch.y

    if model.num_tasks == 1:
        y = y.view(batch_size, -1)
    else:
        y = y.view(batch_size, model.num_tasks)

    if y.shape[1] == 1 and metric == "accuracy":
        y = F.one_hot(torch.squeeze(y, 1), 10)
        
    is_labeled = y == y

    if y.dtype == torch.int64:
        y = y.float()

    if metric in ['accuracy']:
        loss = loss_fn(predictions, y)  
    else:
        if is_labeled.shape == predictions.shape:
            loss = loss_fn(predictions[is_labeled], y.float()[is_labeled])
        else:
            loss = loss_fn(predictions, y)
    if metric == 'accuracy':
        tracking_dict["correct_classifications"] += torch.sum(predictions.argmax(dim=1)== y.argmax(dim=1)).item()

    tracking_dict["y_preds"] += predictions.cpu()
    tracking_dict["y_true"] += y.cpu()
    tracking_dict["batch_losses"].append(loss.item())
    tracking_dict["total_loss"] += loss.item()*batch_size
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
    else:
        raise Exception("Unknown metric name")
    
    return output_dict

def train(model, device, train_loader, optimizer, loss_fct, eval_name, tracker, metric_method):
    """
        Performs one training epoch, i.e. one optimization pass over the batches of a data loader.
    """
    model.train()

    tracking_dict = get_tracking_dict()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = compute_loss_predictions(batch, model, eval_name, device, loss_fct, tracking_dict)

        loss.backward()
        optimizer.step()

        if tracker is not None:
            tracker.log({"Train/BatchLoss": loss.item()})
            
    return compute_final_tracking_dict(tracking_dict, {}, train_loader, eval_name, metric_method=metric_method, train=True)

def eval(model, device, loader, loss_fn, eval_name, metric_method):
    """
        Evaluates a model over all the batches of a data loader.
    """
    model.eval()

    tracking_dict = get_tracking_dict()
    for step, batch in enumerate(loader):
        with torch.no_grad():
            compute_loss_predictions(batch, model, eval_name, device, loss_fn, tracking_dict)

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
