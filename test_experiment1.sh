#!/bin/bash

# 성공한 실험 1 모델들 테스트
successful_datasets=("coupa" "nexus")  # 확실히 성공한 것들
experiment_name="stanford-pedestrian"

echo "🧪 실험 1 모델 테스트 시작"
echo "📅 시작 시간: $(date)"

for dataset in "${successful_datasets[@]}"; do
    echo ""
    echo "="*50
    echo "🔍 테스트 중: $dataset"
    echo "="*50
    
    # 심볼릭 링크를 Pedestrian 데이터로 설정
    rm -f datasets_pedestrian
    ln -sf datasets_experiments/stanford_pedestrian datasets_pedestrian
    
    # 테스트 실행
    python3 test.py \
        --tag $experiment_name-$dataset \
        --n_samples 20 \
        2>&1 | tee logs/test_exp1_${dataset}_$(date +%Y%m%d_%H%M%S).log
    
    echo "✅ 테스트 완료: $dataset"
done

echo ""
echo "🎉 실험 1 테스트 완료!"
echo "📅 종료 시간: $(date)"
