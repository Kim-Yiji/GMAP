# 🚀 서버에서 바로 실행하는 명령어 모음

> 서버 환경이 이미 설정되어 있고 SSH가 연결된 상태에서 바로 사용할 수 있는 명령어들

## ⚡ 빠른 실행 (Copy & Paste)

### 1. 프로젝트 준비
```bash
# 프로젝트 클론 (최초 1회만)
git clone https://github.com/Kim-Yiji/Comjonsul.git
cd Comjonsul  
git checkout CYisSMART

# 의존성 설치 (최초 1회만)
pip install einops  # 추가 패키지만 설치
```

### 2. 빠른 검증
```bash
# Shape 검증 테스트
python demo_final.py
```

### 3. 학습 실행

#### 🏃‍♂️ 기본 학습
```bash
# ETH 데이터셋 학습 (50 에포크)
python train_unified.py \
    --dataset eth \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 1e-4 \
    --obs_len 8 \
    --pred_len 12 \
    --tag "quick_experiment"
```

#### 🚀 고성능 학습
```bash
# 더 큰 모델로 학습
python train_unified.py \
    --dataset eth \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-4 \
    --d_h 256 \
    --d_gp_in 256 \
    --dmrgcn_hidden_dims 128 128 256 256 \
    --agg_method gru \
    --mix_type attention \
    --tag "large_model"
```

#### 🌙 백그라운드 학습 (SSH 끊어져도 계속)
```bash
# nohup으로 백그라운드 실행
nohup python train_unified.py \
    --dataset eth \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 1e-4 \
    --tag "background_training" > training.log 2>&1 &

# 진행상황 확인
tail -f training.log
```

### 4. 테스트 실행
```bash
# 학습된 모델로 테스트
python test_unified.py \
    --dataset eth \
    --model_path ./checkpoints_unified/quick_experiment-eth/eth_best.pth \
    --obs_len 8 \
    --pred_len 12
```

### 5. 모니터링
```bash
# GPU 사용량 실시간 확인
watch -n 1 nvidia-smi

# 학습 로그 실시간 확인  
tail -f training.log

# 실행 중인 Python 프로세스 확인
ps aux | grep python
```

## 🧪 다양한 실험

### Ablation Study
```bash
# Agent path만 사용
python train_unified.py \
    --dataset eth \
    --enable_agent --disable_intra --disable_inter \
    --tag "agent_only"

# Simple head vs GP-Graph head 비교
python train_unified.py --use_simple_head --tag "simple_head"
python train_unified.py --tag "gpgraph_head"
```

### 다른 데이터셋
```bash
# Hotel 데이터셋
python train_unified.py --dataset hotel --tag "hotel_exp"

# 모든 데이터셋 순차 학습
for dataset in eth hotel univ zara1 zara2; do
    python train_unified.py --dataset $dataset --num_epochs 30 --tag "multi_dataset_$dataset"
done
```

### 하이퍼파라미터 튜닝
```bash
# 학습률 변경
python train_unified.py --lr 1e-3 --tag "high_lr"
python train_unified.py --lr 5e-5 --tag "low_lr"

# 배치 사이즈 실험
python train_unified.py --batch_size 32 --tag "large_batch"
python train_unified.py --batch_size 4 --tag "small_batch"
```

## 📊 결과 분석

### 체크포인트 확인
```bash
# 저장된 모델들 확인
ls -la checkpoints_unified/*/

# 최고 성능 모델 찾기
find checkpoints_unified/ -name "*_best.pth"
```

### TensorBoard
```bash
# TensorBoard 실행 (포트 6006)
tensorboard --logdir ./checkpoints_unified --port 6006

# 백그라운드로 실행
nohup tensorboard --logdir ./checkpoints_unified --port 6006 > tensorboard.log 2>&1 &
```

## 🔧 트러블슈팅

### GPU 메모리 부족 시
```bash
# 배치 사이즈 줄이기
python train_unified.py --batch_size 4 --tag "small_memory"

# 작은 모델 사용
python train_unified.py \
    --d_h 64 \
    --dmrgcn_hidden_dims 32 32 64 \
    --tag "lightweight"
```

### 학습 중단/재시작
```bash
# 백그라운드 프로세스 중단
kill $(ps aux | grep "train_unified.py" | awk '{print $2}' | head -1)

# 체크포인트에서 재시작
python train_unified.py \
    --resume ./checkpoints_unified/quick_experiment-eth/checkpoint_epoch_20.pth \
    --tag "resumed_training"
```

## 📋 추천 실행 순서

1. **빠른 검증**: `python demo_final.py`
2. **소규모 테스트**: `python train_unified.py --dataset eth --num_epochs 10 --tag "test"`
3. **본격 학습**: 위의 백그라운드 학습 명령어 사용
4. **성능 평가**: `python test_unified.py` 명령어 사용

---

**💡 Tip**: 모든 명령어를 복사해서 터미널에 바로 붙여넣기하면 됩니다!
