#!/bin/bash
echo "🌅 실험 결과 확인 - $(date)"
echo "="*60

echo "1. 실험 완료 상태:"
if ps aux | grep -q "python3.*train.py"; then
    echo "   🔄 아직 실행 중..."
else
    echo "   ✅ 모든 실험 완료!"
fi

echo ""
echo "2. 성공한 실험 개수:"
ped_count=$(ls checkpoints/ 2>/dev/null | grep "stanford-ped-opt" | wc -l)
all_count=$(ls checkpoints/ 2>/dev/null | grep "stanford-all-opt" | wc -l)
echo "   실험 1 (Pedestrian Only): $ped_count/5개 성공"
echo "   실험 2 (All Classes): $all_count/5개 성공"

echo ""
echo "3. 성공한 모델들 자동 테스트:"
for checkpoint in checkpoints/stanford-*-opt-*; do
    if [ -d "$checkpoint" ]; then
        tag=$(basename "$checkpoint")
        dataset=$(echo "$tag" | sed 's/stanford-.*-opt-//')
        experiment=$(echo "$tag" | grep -o "ped\|all")
        
        echo ""
        echo "🧪 테스트: $dataset ($experiment)"
        
        # 올바른 데이터로 심볼릭 링크 설정
        if [[ "$tag" == *"ped"* ]]; then
            rm -f datasets_pedestrian
            ln -sf datasets_experiments/stanford_pedestrian datasets_pedestrian
        else
            rm -f datasets_pedestrian  
            ln -sf datasets_experiments/stanford_all datasets_pedestrian
        fi
        
        # 테스트 실행
        result=$(python3 test.py --tag "$tag" --n_samples 20 2>/dev/null | grep -E "ADE|FDE")
        echo "   결과: $result"
    fi
done

echo ""
echo "🎉 모든 확인 완료!"
