#!/bin/bash

# Training script for DMRGCN + GP-Graph integration
# Example usage for different datasets

echo "Starting DMRGCN + GP-Graph training..."

# ETH dataset with default group-aware settings
python train.py \
    --dataset eth \
    --gpgraph \
    --group_type euclidean \
    --group_th 2.0 \
    --mix_type mean \
    --enable_agent \
    --enable_intra \
    --enable_inter \
    --share_backbone \
    --obs_len 8 \
    --pred_len 12 \
    --batch_size 32 \
    --num_epochs 128 \
    --lr 1e-4 \
    --tag dmrgcn_gpgraph_eth

echo "Training completed for ETH dataset"

# Hotel dataset with learned grouping
python train.py \
    --dataset hotel \
    --gpgraph \
    --group_type learned \
    --group_th 1.5 \
    --mix_type mlp \
    --enable_agent \
    --enable_intra \
    --enable_inter \
    --share_backbone \
    --st_estimator \
    --obs_len 8 \
    --pred_len 12 \
    --batch_size 32 \
    --num_epochs 128 \
    --lr 1e-4 \
    --tag dmrgcn_gpgraph_hotel

echo "Training completed for Hotel dataset"

echo "All training jobs completed!"
