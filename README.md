# BasicGNNProject
This repository is meant as a starting point for working on your own GNN research projects. This code allows you to tune, train and evaluate basic models (MLP, GCN and GIN) on some well known graph datasets (`ZINC`, `CSL`, and OGB datasets). Some projects based on earlier versions of this repository are: 
- _Weisfeiler and Leman Return with Graph Transformations_, MLG@ECMLPKDD, 2022: [paper](https://openreview.net/pdf?id=Oq5mzL-3SUV), [code](https://github.com/ocatias/WL_Return)
- _Reducing Learning on Cell Complexes to Graphs_, GTRL@ICLR, 2022, [paper](https://openreview.net/pdf?id=HKUxAE-J6lq), [code](https://github.com/ocatias/CellComplexesToGraphs)

If you find this repository helpful please give it a :star:.

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
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
pip install -r requirements.txt
```

### Tracking
Training and different experiments are tracked via [wandb](https://wandb.ai/). If you want to make use of the tracking you need a wandb account. The first time you train a model, you will be prompted to enter you wandb API key. If you want to disable tracking you can do this in the config `Configs/config.yaml`.


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

## Supported Models and Datasets

**Models:** GIN, GCN and MLP. MLP pools all vertex features and then passes the resulting vector through an MLP.

**Datasets:**
- `ZINC` 
- `CSL`: please use cross validation for this dataset
- OGB datasets: `ogbg-molhiv`, `ogbg-moltox21`, `ogbg-molesol`, `ogbg-molbace`, `ogbg-molclintox`, `ogbg-molbbbp`, `ogbg-molsider`, `ogbg-moltoxcast`, `ogbg-mollipo`

## Testing

The integration tests can be executed with
```
python -m unittest
```

## Citations

### Code
```
@inproceedings{neurips-bodnar2021b,
title={Weisfeiler and Lehman Go Cellular: CW Networks},
author={Bodnar, Cristian and Frasca, Fabrizio and Otter, Nina and Wang, Yu Guang and Li{\`o}, Pietro and Mont{\'u}far, Guido and Bronstein, Michael},
booktitle = {Advances in Neural Information Processing Systems},
year={2021}
}
```

```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={International Conference on Learning Representations},
year={2019}
}
```

```
@inproceedings{ogb,
author = {Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
booktitle = {NeurIPS},
title = {{Open Graph Benchmark}: Datasets for Machine Learning on Graphs},
year = {2020}
}
```

### Models
GCN
```
@inproceedings{GCN,
author    = {Thomas N. Kipf and Max Welling},
title     = {Semi-Supervised Classification with Graph Convolutional Networks},
year      = {2017},
booktitle = {ICLR}
}
```

GIN
```
@inproceedings{
xu2018how,
title={How Powerful are Graph Neural Networks?},
author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
booktitle={ICLR},
year={2019}
}
```

### Datasets
ZINC
```
@article{ZINC1,
author = {Gómez-Bombarelli, Rafael and Wei, Jennifer N. and Duvenaud, David and Hernández-Lobato, José Miguel and Sánchez-Lengeling, Benjamín and Sheberla, Dennis and Aguilera-Iparraguirre, Jorge and Hirzel, Timothy D. and Adams, Ryan P. and Aspuru-Guzik, Alán},
title = {Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules},
journal = {ACS Central Science},
year = {2018},
}
```

```
@article{ZINC2,
author = {Sterling, Teague and Irwin, John J.},
title = {ZINC 15 – Ligand Discovery for Everyone},
journal = {Journal of Chemical Information and Modeling},
year = {2015},
}
```

CSL
```
@inproceedings{relational_pooling,
title = {Relational {Pooling} for {Graph} {Representations}},
author = {Murphy, Ryan L and Srinivasan, Balasubramaniam and Rao, Vinayak and Ribeiro, Bruno},
year = {2019},
booktitle = {ICML}
}
```

```
@article{Benchmarking-GNNs,
title={Benchmarking Graph Neural Networks},
author={Dwivedi, Vijay Prakash and Joshi, Chaitanya K and Laurent, Thomas and Bengio, Yoshua and Bresson, Xavier},
journal={arXiv preprint arXiv:2003.00982},
year={2020}
}
```

OGB
```
@inproceedings{ogb,
author = {Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
booktitle = {NeurIPS},
title = {{Open Graph Benchmark}: Datasets for Machine Learning on Graphs},
year = {2020}
}
```
