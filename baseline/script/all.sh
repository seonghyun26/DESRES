cd ../

# CLN025

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --config-name baseline_tda_cln025

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_tica_cln025

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_vde_cln025

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_tae_cln025


# 2JOF
# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     ----config-name baseline_tda_2jof

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_tica_2jof

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_vde_2jof

# CUDA_VISIBLE_DEVICES=$1 python main.py \
#     --config-name baseline_tae_2jof