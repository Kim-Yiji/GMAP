# GMAP: Group-aware Multi-relational Trajectory Prediction

> 25-2 컴종설 프로젝트  
> DMRGCN과 GP-Graph를 통합한 보행자 궤적 예측 모델

## 🎯 프로젝트 개요

GMAP은 **그룹 인식 기반의 다중 관계형 그래프 신경망**을 사용하여 보행자들의 미래 궤적을 예측합니다. 이 프로젝트는 DMRGCN (Disentangled Multi-Relational Graph Convolution Network)과 GP-Graph (Group-aware Pedestrian Graph)를 통합하여 구현되었습니다.

### 주요 특징

- ✨ **3단계 계층적 처리**: 개인 레벨 → 그룹 내 → 그룹 간 상호작용 모델링
- 🔗 **공유 백본 아키텍처**: DMRGCN 백본을 통한 효율적인 특성 추출
- 📊 **다양한 그룹 할당 전략**: 유클리드 거리 기반, 학습 기반 등
- ⚡ **데이터 캐싱**: 5-10배 빠른 학습 속도
- 📈 **경쟁력 있는 성능**: ADE 0.21m, FDE 0.29m (ETH 데이터셋)

## 🚀 빠른 시작

### 1. 설치

```bash
git clone https://github.com/Kim-Yiji/GMAP.git
cd GMAP
git checkout github-upload
pip install -r requirements.txt
```

### 2. 빠른 검증

```bash
python demo_final.py
```

### 3. 학습

```bash
# 기본 학습
python train_unified.py --dataset eth --batch_size 8 --num_epochs 50

# 고성능 설정
python train_unified.py \
    --dataset eth \
    --batch_size 16 \
    --num_epochs 100 \
    --d_h 256 \
    --mix_type attention
```

### 4. 테스트

```bash
python test.py \
    --dataset eth \
    --checkpoint ./checkpoints_unified/eth_best.pth
```

## 📚 상세 문서

- **[README_INTEGRATION.md](README_INTEGRATION.md)**: 모델 아키텍처 및 상세 구현
- **[RUN_COMMANDS.md](RUN_COMMANDS.md)**: 실행 명령어 모음
- **[FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md)**: 테스트 결과 및 성능 분석
- **[BACKUP_GUIDE.md](BACKUP_GUIDE.md)**: 체크포인트 백업 가이드
- **[CACHING_INFO.md](CACHING_INFO.md)**: 데이터 캐싱 시스템 설명

## 📊 성능

| Metric | ETH Dataset | 모델 크기 |
|--------|-------------|----------|
| **ADE** | 0.211 m | 297K params |
| **FDE** | 0.290 m | - |

## 🏗️ 프로젝트 구조

```
GMAP/
├── datasets/          # 데이터 로더 및 전처리
├── model/            # 모델 아키텍처
│   ├── backbone.py           # DMRGCN 백본
│   ├── gpgraph_adapter.py    # 그룹 할당 모듈
│   └── dmrgcn_gpgraph.py     # 통합 모델
├── utils/            # 그래프 유틸리티
├── adapters/         # 호환성 어댑터
├── train_unified.py  # 학습 스크립트
├── test.py           # 테스트 스크립트
└── demo_final.py     # 데모 스크립트
```

## 🛠️ 주요 기능

### 그룹 인식 처리
- 유클리드 거리 기반 그룹 할당
- 학습 기반 그룹 추정
- 시공간 특성 활용

### 다중 관계형 그래프
- 6가지 관계 유형
- 거리 기반 인접 행렬
- 계층적 특성 융합

### 데이터 캐싱
- 자동 전처리 캐싱
- 5-10배 빠른 학습 속도
- 스마트 캐시 관리

## 📝 라이선스

이 프로젝트는 학술 연구 목적으로 개발되었습니다.

## 👥 기여자

25-2 컴종설 팀

---

**💡 Tip**: 자세한 실행 방법은 [RUN_COMMANDS.md](RUN_COMMANDS.md)를 참고하세요!
