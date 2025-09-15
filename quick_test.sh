#!/bin/bash

# Quick Test Script - 학습된 모델 바로 테스트
# Usage: ./quick_test.sh [model_path] [dataset]

MODEL_PATH=${1:-"./checkpoints_unified/quick_exp-eth/eth_best.pth"}
DATASET=${2:-"eth"}

echo "🧪 Quick Test Start"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Available models:"
    find ./checkpoints_unified -name "*.pth" | head -5
    exit 1
fi

echo "📊 Running evaluation..."
python test_unified.py \
    --dataset $DATASET \
    --model_path $MODEL_PATH \
    --obs_len 8 \
    --pred_len 12

echo "✅ Test complete!"
