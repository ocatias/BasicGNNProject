conda activate GNNs
export PYTHONPATH=$PYTHONPATH:$PATH
python Exp/run_model.py --model GIN --dataset CSL --transform_k_wl 3 --k_wl_turbo 0 --batch_size 4 --device CPU --emb_dim 8 --epochs 2
