#!/bin/bash

# Testing script for DMRGCN + GP-Graph integration
# Example usage for evaluation

echo "Starting DMRGCN + GP-Graph testing..."

# Test ETH dataset
python test.py \
    --dataset eth \
    --checkpoint ./checkpoints/dmrgcn_gpgraph_eth-eth/eth_best.pth \
    --obs_len 8 \
    --pred_len 12 \
    --num_samples 20 \
    --motion_analysis \
    --visualize \
    --output_dir ./test_outputs/eth/

echo "Testing completed for ETH dataset"

# Test Hotel dataset  
python test.py \
    --dataset hotel \
    --checkpoint ./checkpoints/dmrgcn_gpgraph_hotel-hotel/hotel_best.pth \
    --obs_len 8 \
    --pred_len 12 \
    --num_samples 20 \
    --motion_analysis \
    --visualize \
    --output_dir ./test_outputs/hotel/

echo "Testing completed for Hotel dataset"

echo "All testing jobs completed!"
