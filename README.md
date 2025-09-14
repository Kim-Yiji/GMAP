# Comjonsul - Integrated DMRGCN + GP-Graph for Pedestrian Trajectory Prediction

25-2 컴종설 프로젝트: DMRGCN과 GP-Graph를 통합한 보행자 궤적 예측 모델

## 📋 프로젝트 개요

이 프로젝트는 두 개의 최신 보행자 궤적 예측 모델을 통합하여 더 강력한 예측 성능을 달성하는 것을 목표로 합니다:

- **DMRGCN (AAAI 2021)**: Multi-Relational Graph Convolution 기반
- **GP-Graph (ECCV 2022)**: Group-based Processing 기반

### 🎯 주요 특징

- **모듈화된 설계**: GP-Graph를 모듈로 분리하여 DMRGCN에 통합
- **추가 Feature**: Local Density, Group Size 등 새로운 특징 추가
- **모션 분석**: 다양한 보행자 모션 타입 분석 및 필터링
- **성능 비교**: 전체 데이터셋 vs 필터링된 데이터셋 성능 비교

## 🏗️ 프로젝트 구조

```
Comjonsul/
├── model/                          # 모델 구현
│   ├── __init__.py
│   ├── dmrgcn.py                   # DMRGCN 원본 구현
│   ├── gpgraph_modules.py          # GP-Graph 모듈화된 컴포넌트
│   ├── social_dmrgcn_gpgraph.py    # 통합 모델
│   ├── predictor.py                # 예측기
│   ├── loss.py                     # 손실 함수
│   └── ...
├── utils/                          # 유틸리티 함수
│   ├── dataloader.py              # 데이터 로더
│   ├── augmentor.py               # 데이터 증강
│   └── visualizer.py              # 시각화
├── datasets/                       # 데이터셋 (ETH/UCY)
│   ├── eth/
│   ├── hotel/
│   ├── univ/
│   ├── zara1/
│   └── zara2/
├── checkpoints/                    # 모델 체크포인트
├── train_integrated.py            # 통합 모델 훈련
├── test_integrated.py             # 통합 모델 테스트
├── analyze_motions.py             # 모션 분석
├── create_filtered_dataset.py     # 필터링된 데이터셋 생성
├── compare_performance.py         # 성능 비교
└── README.md                      # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install torch torchvision numpy matplotlib tqdm tensorboard
```

### 2. 데이터셋 준비

ETH/UCY 데이터셋이 `datasets/` 폴더에 준비되어 있어야 합니다.

### 3. 통합 모델 훈련

```bash
# 기본 훈련 (ETH 데이터셋)
python train_integrated.py --dataset eth --tag integrated-dmrgcn-gpgraph

# 모든 기능 활성화하여 훈련
python train_integrated.py \
    --dataset eth \
    --tag integrated-dmrgcn-gpgraph \
    --use_group_processing \
    --use_density \
    --use_group_size \
    --num_epochs 128
```

### 4. 모델 테스트

```bash
# 훈련된 모델 테스트
python test_integrated.py --tag integrated-dmrgcn-gpgraph --dataset eth
```

## 📊 모션 분석 및 데이터셋 필터링

### 1. 모션 분석

```bash
# ETH 데이터셋의 모션 타입 분석
python analyze_motions.py --dataset eth --data_split train --save_plots

# 결과: 모션 분포 차트와 특징 분석
```

### 2. 필터링된 데이터셋 생성

```bash
# 특정 모션 타입만 포함하는 데이터셋 생성
python create_filtered_dataset.py \
    --source_dataset eth \
    --motion_types linear curved direction_change \
    --create_all_combinations

# 결과: datasets_filtered/ 폴더에 다양한 모션 조합 데이터셋 생성
```

### 3. 성능 비교

```bash
# 전체 데이터셋 vs 필터링된 데이터셋 성능 비교
python compare_performance.py \
    --model_tag integrated-dmrgcn-gpgraph \
    --dataset eth \
    --compare_filtered \
    --save_plots
```

## 🔧 모델 파라미터

### DMRGCN 파라미터
- `--n_stgcn`: GCN 레이어 수 (기본값: 1)
- `--n_tpcnn`: CNN 레이어 수 (기본값: 4)
- `--kernel_size`: 커널 크기 (기본값: 3)
- `--output_size`: 출력 특징 차원 (기본값: 5)

### GP-Graph 파라미터
- `--group_d_type`: 그룹 거리 타입 ('learned_l2norm', 'learned', 'euclidean')
- `--group_d_th`: 그룹 거리 임계값 ('learned', float)
- `--group_mix_type`: 그룹 혼합 타입 ('mlp', 'cnn', 'mean', 'sum')
- `--use_group_processing`: 그룹 기반 처리 활성화

### 추가 특징 파라미터
- `--density_radius`: 밀도 계산 반경 (기본값: 2.0)
- `--group_size_threshold`: 최소 그룹 크기 임계값 (기본값: 2)
- `--use_density`: 지역 밀도 특징 사용
- `--use_group_size`: 그룹 크기 특징 사용

## 📈 모션 타입

모델이 처리할 수 있는 보행자 모션 타입:

1. **Linear Motion**: 직선 궤적, 최소한의 방향 변화
2. **Curved Motion**: 비선형 궤적, 부드러운 곡선
3. **Direction Change**: 상당한 방향 변화가 있는 궤적
4. **Group Motion**: 고속, 조율된 그룹 움직임
5. **Stationary**: 최소한의 움직임 궤적

## 📊 성능 지표

- **ADE (Average Displacement Error)**: 모든 시간 단계의 평균 오차
- **FDE (Final Displacement Error)**: 최종 예측 시간 단계의 오차
- **Loss**: 궤적 예측을 위한 다변량 가우시안 손실

## 🎯 사용 예시

### 전체 워크플로우

```bash
# 1. 모션 분석
python analyze_motions.py --dataset eth --data_split train --save_plots

# 2. 필터링된 데이터셋 생성
python create_filtered_dataset.py --source_dataset eth --create_all_combinations

# 3. 통합 모델 훈련
python train_integrated.py --dataset eth --use_group_processing --use_density --use_group_size

# 4. 성능 비교
python compare_performance.py --model_tag integrated-dmrgcn-gpgraph --dataset eth --compare_filtered
```

### 특정 모션만 테스트

```bash
# 선형 모션만 포함하는 데이터셋으로 테스트
python compare_performance.py \
    --model_tag integrated-dmrgcn-gpgraph \
    --dataset eth \
    --compare_filtered
```

## 📚 참고 문헌

```bibtex
@article{bae2021dmrgcn,
  title={Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}

@inproceedings{bae2022gpgraph,
  title={Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

## 🤝 기여

이 프로젝트는 25-2 컴종설 수업의 일환으로 개발되었습니다.

## 📄 라이선스

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.
