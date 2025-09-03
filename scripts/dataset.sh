cd ../dataset

CUDA_VISIBLE_DEVICES=$1 python create_dataset.py \
    --molecule=NTL9 \
    --dataset_size=50000 \
    --simulation_idx=0
