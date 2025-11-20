#!/bin/bash

# DMRGCN 디렉토리로 이동
cd /raid/guest/OATMeal_Queens/newbie/yoonhee/DMRGCN

# SDD 데이터셋 폴더 목록
datasets=(
    # "sdd_bookstore"
    "sdd_coupa"
    # "sdd_deathCircle"
    # "sdd_hyang"
    # "sdd_little"
    # "sdd_nexus"
)

# 로그 디렉토리 생성
mkdir -p logs

# 각 데이터셋에 대해 train.py 실행
for dataset in "${datasets[@]}"; do
    echo "Starting training for $dataset..."
    
    CUDA_VISIBLE_DEVICES=2 nohup python -u train.py \
        --dataset $dataset \
        --use_cache \
        --train_cache /raid/guest/SDD_2beon/$dataset/train/preproc_cache_obs8_pred12_skip1.pt \
        --val_cache   /raid/guest/SDD_2beon/$dataset/val/preproc_cache_obs8_pred12_skip1.pt \
        --obs_seq_len 8 \
        --pred_seq_len 12 \
        --batch_size 128 \
        --num_epochs 32 \
        --tag 32-${dataset}-cached > logs/32_${dataset}_training.log 2>&1 &
    
    
    # GPU 메모리 정리를 위해 약간의 대기 시간 (선택사항)
    sleep 2
done

echo "All training jobs have been started!"
echo "Check logs/ directory for output files."
echo "To monitor running processes: ps aux | grep train.py"







