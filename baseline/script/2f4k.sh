cd ../

# 2f4k

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name baseline_tda_2f4k

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_tica_2f4k

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_vde_2f4k

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_tae_2f4k
