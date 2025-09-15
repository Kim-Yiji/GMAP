#!/bin/bash

# Testing script for Unified Trajectory Prediction Model
# Usage: ./run_test.sh [dataset] [tag] [samples]

DATASET=${1:-eth}
TAG=${2:-unified-model}
SAMPLES=${3:-20}

echo "Starting testing with:"
echo "Dataset: $DATASET"
echo "Tag: $TAG"
echo "Samples: $SAMPLES"

python test.py \
    --dataset $DATASET \
    --tag $TAG \
    --n_samples $SAMPLES \
    --n_trials 100 \
    --visualize

echo "Testing completed!"
echo "Results saved in checkpoints/${TAG}-${DATASET}/"
