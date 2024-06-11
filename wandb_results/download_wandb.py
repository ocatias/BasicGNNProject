import pandas as pd
import wandb

api = wandb.Api(timeout=19)

# Project is specified by <entity/project-name>
# runs = api.runs("obrichta-diploma-thesis/k-WL-turbo", filters={"created_at": {'$gte': "2024-01-19 00:00:00"}})
runs = api.runs("obrichta-diploma-thesis/k-gnn-original")
l = []
summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)
    # print(run.config.items())
    # print(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k, v in run.config.items()
         if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    d = dict()
    d.update({k: v for k, v in run.config.items()
              if not k.startswith('_')})
    d.update(run.summary._json_dict)
    l.append(d)
runs_df = pd.DataFrame(l)

runs_df.to_csv("project_original.csv")
# runs_df.to_csv("project_my.csv")
