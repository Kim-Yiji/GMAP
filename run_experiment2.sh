#!/bin/bash

# 성공한 데이터셋만 실험 (gates, little 제외)
datasets=("bookstore" "coupa" "deathCircle" "hyang" "nexus")
gpu_id=3
experiment_name="stanford-all"

echo "🚀 실험 2 시작: All Classes"
echo "📅 시작 시간: $(date)"
echo "📊 총 데이터셋 수: ${#datasets[@]}"

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    current=$((i+1))
    
    echo ""
    echo "="*60
    echo "🔄 [$current/${#datasets[@]}] 현재 실험: $dataset (All Classes)"
    echo "⏰ 시작 시간: $(date)"
    echo "="*60
    
    # 심볼릭 링크를 All Classes 데이터로 설정
    rm -f datasets_pedestrian
    ln -sf datasets_experiments/stanford_all datasets_pedestrian
    
    # 로그 파일명
    log_file="logs/exp2_${dataset}_$(date +%Y%m%d_%H%M%S).log"
    
    # 학습 실행
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -u train.py \
        --dataset $dataset \
        --tag $experiment_name-$dataset \
        --num_epochs 256 \
        --batch_size 128 \
        --lr 0.0001 \
        --obs_seq_len 2 \
        --pred_seq_len 3 \
        --use_lrschd \
        2>&1 | tee $log_file
    
    echo "✅ [$current/${#datasets[@]}] 완료: $dataset"
    echo "📄 로그 파일: $log_file"
done

echo ""
echo "🎉 실험 2 전체 완료!"
echo "📅 종료 시간: $(date)"
