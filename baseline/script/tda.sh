cd ../

molecule=${2:-"1fme"}
CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name baseline_tda_$molecule