conda activate GNNs
export PYTHONPATH=$PYTHONPATH:$PATH
python Exp/run_model.py --model GIN --dataset CSL --transform_k_wl 3 --k_wl_turbo 0 --batch_size 4 --device CPU --emb_dim 8 --epochs 2
python Exp/run_model.py --model GIN --dataset ZINC --transform_k_wl 3 --k_wl_turbo 0 --batch_size 4 --emb_dim 16 --epochs 2
python Exp/run_experiment.py -grid Configs/Benchmark/GIN_grid.yaml -dataset ogbg-moltox21 --candidates 100 --repeats 1