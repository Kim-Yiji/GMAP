#!/bin/bash

# Quick Training Script - 서버에서 바로 실행
# Usage: ./quick_train.sh [dataset] [epochs] [batch_size] [tag]

# Default values
DATASET=${1:-"eth"}
EPOCHS=${2:-50}
BATCH_SIZE=${3:-8}
TAG=${4:-"quick_exp"}

echo "🚀 Quick Training Start"
echo "Dataset: $DATASET | Epochs: $EPOCHS | Batch: $BATCH_SIZE | Tag: $TAG"
echo ""

# Quick validation first
echo "🧪 Running validation test..."
python demo_final.py | tail -5
if [ $? -ne 0 ]; then
    echo "❌ Validation failed!"
    exit 1
fi

echo ""
echo "🏃‍♂️ Starting training..."

# Run training
python train_unified.py \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --lr 1e-4 \
    --obs_len 8 \
    --pred_len 12 \
    --tag $TAG \
    --log_interval 10 \
    --save_interval 5

echo "✅ Training complete! Check: ./checkpoints_unified/${TAG}-${DATASET}/"
