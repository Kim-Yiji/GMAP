#!/bin/bash

# DMRGCN 디렉토리로 이동
cd /raid/guest/OATMeal_Queens/newbie/yoonhee/DMRGCN

# SDD 데이터셋 폴더 목록 (학습한 것들)
datasets=(
    "sdd_bookstore"
    "sdd_coupa"
    "sdd_deathCircle"
    #"sdd_hyang"
    "sdd_nexus"
)

# 로그 디렉토리 생성
mkdir -p logs

# 각 데이터셋에 대해 test.py 실행
for dataset in "${datasets[@]}"; do
    tag="1118-${dataset}-cached"
    
    # 체크포인트 존재 여부 확인
    checkpoint_path="./checkpoints/${tag}/${dataset}_best.pth"
    if [ ! -f "$checkpoint_path" ]; then
        echo "Warning: Checkpoint not found for $tag: $checkpoint_path"
        echo "Skipping $dataset..."
        continue
    fi
    
    echo "Starting testing for $dataset (tag: $tag)..."
    
    CUDA_VISIBLE_DEVICES=2 python -u test.py \
        --tag $tag \
        --use_cache \
        --test_cache /raid/guest/SDD_2beon/$dataset/test/preproc_cache_obs8_pred12_skip1.pt \
        --n_samples 20 > logs/1118_${dataset}_test.log 2>&1
    
    echo "Completed testing for $dataset"
    echo "Results saved to: checkpoints/${tag}/results.txt"
    echo "---"
done

echo "All testing jobs have been completed!"
echo "Check logs/ directory for output files."
echo "Check checkpoints/*/results.txt for ADE/FDE results."



