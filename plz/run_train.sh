#!/bin/bash

# Training script for Unified Trajectory Prediction Model
# Usage: ./run_train.sh [dataset] [tag] [epochs]

DATASET=${1:-eth}
TAG=${2:-unified-model}
EPOCHS=${3:-300}

echo "Starting training with:"
echo "Dataset: $DATASET"
echo "Tag: $TAG" 
echo "Epochs: $EPOCHS"

python train.py \
    --dataset $DATASET \
    --tag $TAG \
    --num_epochs $EPOCHS \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --n_stgcn 1 \
    --n_tpcnn 4 \
    --kernel_size 3 \
    --obs_seq_len 8 \
    --pred_seq_len 12 \
    --d_type velocity_aware \
    --mix_type attention \
    --include_velocity \
    --include_acceleration \
    --use_lrschd \
    --visualize \
    --save_freq 10 \
    --clip_grad 10.0

echo "Training completed!"
