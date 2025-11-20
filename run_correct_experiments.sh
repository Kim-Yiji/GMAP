#!/bin/bash

echo "🚀 올바른 Stanford 실험 시작"
echo "📅 시작 시간: $(date)"

# 성공 가능성 높은 데이터셋들
datasets=("bookstore" "coupa" "deathCircle" "hyang" "nexus")

# 실험 1: Pedestrian Only
echo ""
echo "="*60
echo "🔬 실험 1: Pedestrian Only (obs_len=1, pred_len=1)"
echo "="*60

# 심볼릭 링크를 Pedestrian 데이터로 설정
rm -f datasets_pedestrian
ln -sf datasets_experiments/stanford_pedestrian datasets_pedestrian
echo "📂 datasets_pedestrian -> stanford_pedestrian 연결됨"

for dataset in "${datasets[@]}"; do
    echo ""
    echo "🔄 실험 1-$dataset 시작: $(date)"
    
    CUDA_VISIBLE_DEVICES=2 python3 -u train.py \
        --dataset $dataset \
        --tag stanford-ped-opt-$dataset \
        --num_epochs 256 \
        --batch_size 16 \
        --lr 0.0001 \
        --obs_seq_len 1 \
        --pred_seq_len 1 \
        --use_lrschd \
        2>&1 | tee logs/opt_exp1_${dataset}_$(date +%Y%m%d_%H%M%S).log
    
    echo "✅ 실험 1-$dataset 완료: $(date)"
    
    # 체크포인트 확인
    if [ -d "checkpoints/stanford-ped-opt-$dataset" ]; then
        echo "   ✅ 체크포인트 생성됨"
    else
        echo "   ❌ 체크포인트 생성 실패"
    fi
done

# 실험 2: All Classes  
echo ""
echo "="*60
echo "🔬 실험 2: All Classes (obs_len=1, pred_len=1)"
echo "="*60

# 심볼릭 링크를 All Classes 데이터로 변경
rm -f datasets_pedestrian
ln -sf datasets_experiments/stanford_all datasets_pedestrian
echo "📂 datasets_pedestrian -> stanford_all 연결됨"

for dataset in "${datasets[@]}"; do
    echo ""
    echo "🔄 실험 2-$dataset 시작: $(date)"
    
    CUDA_VISIBLE_DEVICES=3 python3 -u train.py \
        --dataset $dataset \
        --tag stanford-all-opt-$dataset \
        --num_epochs 256 \
        --batch_size 16 \
        --lr 0.0001 \
        --obs_seq_len 1 \
        --pred_seq_len 1 \
        --use_lrschd \
        2>&1 | tee logs/opt_exp2_${dataset}_$(date +%Y%m%d_%H%M%S).log
    
    echo "✅ 실험 2-$dataset 완료: $(date)"
    
    # 체크포인트 확인
    if [ -d "checkpoints/stanford-all-opt-$dataset" ]; then
        echo "   ✅ 체크포인트 생성됨"
    else
        echo "   ❌ 체크포인트 생성 실패"
    fi
done

echo ""
echo "🎉 모든 최적화 실험 완료!"
echo "📅 종료 시간: $(date)"

# 최종 결과 요약
echo ""
echo "📊 실험 결과 요약:"
echo "실험 1 (Pedestrian Only) 성공:"
ls checkpoints/ | grep "stanford-ped-opt" | wc -l | awk '{print "   " $1 "개 성공"}'

echo "실험 2 (All Classes) 성공:"  
ls checkpoints/ | grep "stanford-all-opt" | wc -l | awk '{print "   " $1 "개 성공"}'
