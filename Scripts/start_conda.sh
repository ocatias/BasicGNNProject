conda activate GNNs
export PYTHONPATH=$PYTHONPATH:$PATH
python Exp/run_model.py --model GIN --dataset CSL --transform_k_wl 3 --k_wl_turbo 0 --batch_size 4 --device CPU --emb_dim 8 --epochs 2 --num_layer 5 --num_mlp_layers 5
python Exp/run_model.py --model GIN --dataset ogbg-moltox21 --transform_k_wl 3 --k_wl_turbo 0 --batch_size 4 --emb_dim 16 --epochs 2 --k_wl_separate_embedding 1 --k_wl_pool_function cat
python Exp/run_experiment.py -grid Configs/Benchmark/GIN_grid.yaml -dataset ogbg-moltox21 --candidates 100 --repeats 1 --device 0
python Exp/run_model.py --model GIN --dataset IMDB-BINARY --transform_k_wl 3 --k_wl_turbo 0 --batch_size 4 --emb_dim 16 --epochs 100 --k_wl_separate_embedding 0 --k_wl_pool_function mode --device 0 --k_wl_set_based 1
python Exp/run_model.py --model GIN --dataset IMDB-BINARY --transform_k_wl 1 --k_wl_turbo 1 --batch_size 4 --emb_dim 16 --epochs 100 --k_wl_separate_embedding 0 --k_wl_pool_function mode --device 0 --k_wl_set_based 1
python Exp/run_experiment.py -grid Configs/Benchmark/GIN_grid.yaml -dataset IMDB-BINARY --candidates 100 --repeats 1 --device 0 >> print_out.txt

python Exp/run_model.py --model GIN --dataset ogbg-moltox21 --transform_k_wl 3 --k_wl_turbo 1 --batch_size 8 --emb_dim 16 --epochs 2 --k_wl_pool_function mode --k_wl_set_based 1 --sequential_k_wl 1 --connected_k_wl_last_k 1 --filter_data_max_graph_size 70
