# BasicGNNProject
This repository is meant as a starting point for your own GNN research projects. This code allows you to tune, train and evaluate basic models on well known graph datasets. 

Projects based on this repository:
- _Expressivity-Preserving GNN Simulation_, NeurIPS, 2023: [paper](https://openreview.net/forum?id=ytTfonl9Wd), [code](https://github.com/ocatias/GNN-Simulation)
- _Expectation-Complete Graph Representations with Homomorphisms_, ICML, 2023: [paper](https://openreview.net/forum?id=ppgRPC14uI), [code](https://github.com/ocatias/HomCountGNNs)
- _Weisfeiler and Leman Return with Graph Transformations_, MLG@ECMLPKDD, 2022: [paper](https://openreview.net/pdf?id=Oq5mzL-3SUV), [code](https://github.com/ocatias/WL_Return)
- _Reducing Learning on Cell Complexes to Graphs_, GTRL@ICLR, 2022, [paper](https://openreview.net/pdf?id=HKUxAE-J6lq), [code](https://github.com/ocatias/CellComplexesToGraphs)

If you find this repository helpful please give it a :star:.

## Supported Models and Datasets

**Models:**
- Message Pasing Graph Neural Networks: `GIN`, `GCN`, `GAT`
- Equivariant Subgraph Aggregation Networks: `DS`, `DSS`
- Multilayer perceptron that ignores the graph structure: `MLP`

**Datasets:**
- `ZINC` 
- `CSL`: please use cross validation for this dataset
- OGB datasets: `ogbg-molhiv`, `ogbg-moltox21`, `ogbg-molesol`, `ogbg-molbace`, `ogbg-molclintox`, `ogbg-molbbbp`, `ogbg-molsider`, `ogbg-moltoxcast`, `ogbg-mollipo`
- Long Range Graph Benchmark datasets: `Peptides-struct`, `Peptides-func`, `PascalVOC-SP`
- QM9: `QM9` or `QM9_i` if you only want to predict the i-th property


## Setup

Clone this repository and open the directory
```
git clone https://github.com/ocatias/BasicGNNProject
cd BasicGNNProject
```

Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

Create a conda environment (this assume miniconda is installed)
```
conda create --name GNNs
```

Activate environment
```
conda activate GNNs
```

Install dependencies
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch; conda install -c pyg pyg=2.2.0; pip install -r requirements.txt
```

### Tracking
Per default, experiments are tracked tracked via [wandb](https://wandb.ai/). This can be disabled in `Configs/config.yaml`. If you want to make use of this tracking you need a wandb account. The first time you train a model, you will be prompted to enter you wandb API key. If you want to disable tracking you can do this in the config `Configs/config.yaml`.

## How to Train a GNN

### Running a Model

To train a GNN `$GNN` once on a datasets `$dataset` run
```
python Exp/run_model.py --model $GNN --dataset $dataset
```

For example `python Exp/run_model.py --model GIN --dataset ZINC`. This trains the GNN GIN on the ZINC dataset a single time. The result of the training will be shown in the terminal. The different hyperparameters of the GNN can be set via commandline parameters. For more details call `python Exp/run_model.py -h`.

### Running a Series of Experiments

The script `Exp/run_experiment.py` optimizes hyperparameters over a parameter grid and then evaluates the parameters with the best performance on the validation set multiple times. For example:
```
python Exp/run_experiment.py -grid Configs/Benchmark/GIN_grid.yaml -dataset ogbg-molesol --candidates 20 --repeats 10 
```
This command tries 20 hyperparameter configurations defined in the `GIN_grid.yaml` config on the `ogbg-molesol` dataset and evaluates the best parameters 10 times. The result of these experiments will be stored in the directory `Results/ogbg-molesol_GIN_grid.yaml`, the averages of the best parameters are stored in `final.json`. If you have a dataset that requires cross-validation (e.g. `CSL`), then you need to set the number of folds (for example `--folds 10`).

### Tuning Hyperparameters with WandB

As `Exp/run_model.py` allows to set model hyperparameters from the commandline, we can use WandB sweeps to optimize hyperparameters.  Here is a short guide, you need to specify your parameter and scripts to run in a config file (see `Configs/WandB_grids/example_grid.yaml`). The sweep can then be initialized with
```
wandb sweep Configs/WandB_Grids/example_grid.yaml
```
This command will tell you the command needed to join agents to the sweep. You can even join agents on different computers to the same sweep! Sweeps can also be initialized purely from scripts. More details on sweeps be found [here](https://wandb.ai/site/sweeps).



## Testing

To run integration tests
```
python -m unittest
```

## Citations
**Models**
- GIN: _How Powerful are Graph Neural Networks?_; Xu et al.; ICLR 2019
- GCN: _Semi-Supervised Classification with Graph Convolutional Networks_; Kipf and Welling; ICLR 2017
- GAT: _Graph Attention Networks_; Veličković at al.;  ICLR 2018
- DS and DSS: _Equivariant Subgraph Aggregation Networks_; Bevilacqua et al.; ICLR 2022


**Datasets**
- ZINC: _Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules_; Gómez-Bombarelli et al.; ACS Central Science 2018
- ZINC: _ZINC 15 – Ligand Discovery for Everyone_; Sterling and Irwin; Journal of Chemical Information and Modeling 2018
- CSL: _Relational Pooling for Graph Representations_; Murphy et al.; ICML 2019
- OGB: _Open Graph Benchmark: Datasets for Machine Learning on Graph_; Hu et al.; NeurIPS 2020
- Long Range Graph Benchmark: _Long Range Graph Benchmark_; Dwivedi et al.; NeurIPS 2022
- QM9: _MoleculeNet: A Benchmark for Molecular Machine Learning_; Wu et al.; Chemical Science 2018