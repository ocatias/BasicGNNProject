import os
import glob
import yaml
import math
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str)
path = parser.parse_args().path

models = ["gin", "gin+hom", "gcn", "gcn+hom", "gin+F", "gin+F+hom", "gcn+F", "gcn+F+hom"]
datasets1 = ["ogbg-molbace", "ogbg-molclintox", "ogbg-molbbbp", "ogbg-molsider", "ogbg-moltoxcast", ]
datasets2 = ["ogbg-mollipo", "ogbg-molhiv", "ogbg-moltox21", "ogbg-molesol", "ZINC"]

datasets = datasets1 + datasets2

scoring = {"ogbg-molbace": "roc-auc", "ogbg-molclintox": "roc-auc", "ogbg-molbbbp": "roc-auc",
"ogbg-molsider": "roc-auc", "ogbg-moltoxcast": "roc-auc", "ogbg-mollipo": "rmse",
"ogbg-molhiv": "roc-auc", "ogbg-moltox21": "roc-auc", "ogbg-molesol": "rmse", "ZINC": "mae"}

def get_baseline(model):
    if "+F" in model:
        return model[0:5]
    else:
        return model[0:3]

runtime_hours = 0

df_avg = pd.DataFrame(columns=datasets, index=models)
df_std = pd.DataFrame(columns=datasets, index=models)

df_full_latex = pd.DataFrame(columns=datasets, index=models)

for computer_dir in glob.glob(os.path.join(path, "*")):
    for experiment in glob.glob(os.path.join(computer_dir, "*")):
        file_name = os.path.split(experiment)[-1]
        dataset = file_name.split("_")[0]
        model_full = "_".join(file_name.split("_")[1:])
        model = model_full.split("_")[0]

        uses_features = "with_features" in file_name
        uses_hom = ".overflow_filtered" in file_name

        results_path = os.path.join(experiment, "final.json")
        if f"ogbg-{dataset.lower()}" in datasets:
            dataset = f"ogbg-{dataset.lower()}"
        if dataset not in datasets:
            continue
        if model not in models:
            continue

        if not os.path.exists(results_path):
            continue

        model = model + ("+F" if uses_features else "") + ("+hom" if uses_hom else "")

        print(f"{model}: {dataset}, {uses_features}, {uses_hom}, \t{model}")

        with open(results_path) as file:
            results = yaml.safe_load(file)

        if scoring[dataset] == "roc-auc":
            df_avg[dataset][model] = round(results["test-avg"]*100, 1)
            df_std[dataset][model] = math.ceil(results["test-std"]*1000) / 10
        else:
            df_avg[dataset][model] = round(results["test-avg"], 3)
            df_std[dataset][model] = math.ceil(results["test-std"]*1000) /1000

print(df_avg)

for dataset in datasets:
    for model in models:
        if not np.isnan(df_avg[dataset][model]):

            is_better = False
            if "hom" in model:
                curr = df_avg[dataset][model]
                baseline = df_avg[dataset][get_baseline(model)]
                is_better = curr > baseline if scoring[dataset] == "roc-auc" else curr < baseline


            if not is_better or np.isnan(df_avg[dataset][get_baseline(model)]):
                df_full_latex[dataset][model] = f"${df_avg[dataset][model]} \pm {df_std[dataset][model]}$"
            else:
                df_full_latex[dataset][model] = "$\mathbf{" + f"{df_avg[dataset][model]} \pm {df_std[dataset][model]}" + "}$"

        else:
            df_full_latex[dataset][model] = ""

print(df_full_latex)

df_full_latex_1 = df_full_latex[datasets1].copy()
df_full_latex_2 = df_full_latex[datasets2].copy()







df_combined = pd.DataFrame(columns=["Top 1 / 2 / 3", "Better than baseline", "At least as good as baseline"], index=models)
df_ranks = df_avg.copy()
for dataset in datasets1 + datasets2:
    print(dataset)
    df_ranks[dataset] = df_ranks[dataset].rank(method='min', ascending=scoring[dataset] != "roc-auc")

print(df_ranks)

get_top_k = lambda k, ranks: int(len(list(filter(lambda x: x <= k, ranks))))
get_nr_times_better = lambda me, comparison: len(list(filter(lambda x: x[0] < x[1], zip(me, comparison))))
get_nr_times_betterequal = lambda me, comparison: len(list(filter(lambda x: x[0] <= x[1], zip(me, comparison))))

for model in models:
    ranks = df_ranks.loc[model]

    top_k = [get_top_k(k, ranks) for k in range(1,4)]
    print(model, top_k)
    better_than_baseline, better_than_amp = 0, 0

    better_than_baseline = get_nr_times_better(ranks, df_ranks.loc[get_baseline(model)])
    better_than_baseline_or_eq = get_nr_times_betterequal(ranks, df_ranks.loc[get_baseline(model)])

    print(better_than_baseline)

    top_str = r""
    for i in range(3):
        top_str += rf"{top_k[i]*10}\%" if top_k[i] !=  00 else r"\phantom{0}0\%"
        if i < 2:
            top_str += " / "
    df_combined["Top 1 / 2 / 3"][model] = top_str


    if model == get_baseline(model):
        df_combined["Better than baseline"][model]  = "-"
        df_combined["At least as good as baseline"][model]  = "-"
    else:
        df_combined["Better than baseline"][model] = f"{better_than_baseline*10}\%"
        df_combined["At least as good as baseline"][model] = f"{better_than_baseline_or_eq*10}\%"

with open(os.path.join(".", f"full_tables.txt"), "w") as file:
    file.write(df_full_latex_1.style.to_latex())
    file.write(df_full_latex_2.style.to_latex())
    file.write(df_combined.style.to_latex())
