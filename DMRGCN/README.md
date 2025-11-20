# DMRGCN: Disentangled Multi-Relational Graph Convolutional Network

<h2 align="center">Disentangled Multi-Relational Graph Convolutional Network for<br>Pedestrian Trajectory Prediction</h2>
<p align="center">
  <a href="https://InhwanBae.github.io/"><strong>Inhwan Bae</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ"><strong>Hae-Gon Jeon</strong></a>
  <br>
  AAAI 2021
</p>

## 📋 Table of Contents

- [프로젝트 개요](#프로젝트-개요)
- [주요 확장 사항](#주요-확장-사항)
- [빠른 시작 (Quick Start)](#빠른-시작-quick-start)
- [데이터셋 설정](#데이터셋-설정)
- [모델 학습](#모델-학습)
- [Threat Score 시각화](#threat-score-시각화)
- [프로젝트 구조](#프로젝트-구조)
- [상세 문서](#상세-문서)
- [문제 해결](#문제-해결)

> 💡 **새로운 사용자**: [README_INDEX.md](./README_INDEX.md)에서 전체 문서 구조를 확인하세요.

---

## 프로젝트 개요

이 프로젝트는 DMRGCN (Disentangled Multi-Relational Graph Convolutional Network)을 기반으로 하며, **SDD (Stanford Drone Dataset)에 대한 4-relation extension**과 **threat score 시각화 기능**을 추가했습니다.

### 원본 DMRGCN 기능
- **Disentangling social interaction**: 고차원 사회 관계에서의 over-smoothing 및 biased weighting 문제 해결
- **Disentangled Multi-scale Aggregation**: 가중 그래프에서 더 나은 사회적 상호작용 표현
- **Global Temporal Aggregation**: 보행자가 방향을 변경할 때 누적 오류 완화
- **DropEdge**: 관계 엣지를 무작위로 제거하여 over-fitting 방지

### 주요 확장 사항
- **4-relation extension**: 2개에서 4개의 relation으로 확장
  - R_PP^disp: 보행자-보행자 상대 변위 크기
  - R_PP^dist: 보행자-보행자 거리 (미터)
  - R_PP^threat: 보행자-보행자 위협 점수
  - R_PO^threat: 보행자-동적 객체 위협 점수
- **Threat score 계산**: Affordance feature 기반 위협 점수 계산
- **SDD 데이터셋 지원**: Stanford Drone Dataset 지원
- **Threat score 시각화**: 특정 인물에 대한 위협 점수를 영상에 시각화

---

## 주요 확장 사항

### 1. 4-Relation Extension

기존 2-relation (displacement, distance)에서 4-relation으로 확장:

```python
# 4 relations
split = [
    [0, 1/4, 2/4, 3/4, 1],      # R_PP^disp: displacement magnitude bins
    [0, 1/2, 1, 2, 4],          # R_PP^dist: distance bins (meters)
    [0.2, 0.4, 0.6, 0.8],       # R_PP^threat: PP threat score bins
    [0.2, 0.4, 0.6, 0.8],       # R_PO^threat: PO threat score bins
]
```

### 2. Threat Score 계산

Affordance feature 기반 위협 점수:

```python
z_ij = [
    d_ij,      # distance
    v^+_ij,    # approach speed (positive component)
    size_j,    # obstacle size (bbox area)
    ttc_ij     # time-to-collision
]
T_ij = sigmoid(w · normalize(z_ij))
```

### 3. SDD 데이터셋 지원

- ETH/UCY 형식 데이터 지원
- 라벨 CSV 지원 (track_id → class)
- 클래스별 크기 CSV 지원
- 자동 형식 감지 (SDD 원본 형식 및 ETH/UCY 형식)

---

## 빠른 시작 (Quick Start)

### 1. 환경 설정

```bash
# Python 3.7+, PyTorch 1.6.0+
pip install torch torchvision numpy matplotlib opencv-python seaborn tqdm
```

### 2. 데이터셋 준비

**전처리된 SDD 데이터셋 사용** (권장):
```bash
# 데이터 경로
/raid/guest/SDD_2beon/sdd_bookstore/
  ├── train/
  ├── val/
  └── test/
```

**데이터 형식**: ETH/UCY 형식 (탭 구분)
```
frame_id    track_id    x    y
0           12          43.1  16.2
0           43          41.2  21.8
```

### 3. 모델 학습 (선택사항)

모델을 학습하지 않고도 threat score 시각화는 가능합니다. 학습이 필요하면:

```bash
cd /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN

python train.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/train \
    --obs_seq_len 8 --pred_seq_len 12 \
    --n_stgcn 1 --n_tpcnn 4 --kernel_size 3 \
    --batch_size 64 --num_epochs 80 \
    --lr 1e-4 --use_lrschd --lr_sh_rate 32 \
    --tag dmrgcn_4rel_sdd \
    --compute_metrics --kstpes 20
```

### 4. Threat Score 시각화 (가장 빠른 방법)

**스크립트 사용** (가장 간단):
```bash
cd /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN

# Track ID 20으로 시각화
./visualize_bookstore_video0.sh 20

# 다른 Track ID
./visualize_bookstore_video0.sh 5
```

**Python 명령어 직접 사용**:
```bash
python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/test \
    --target_track_id 20 \
    --sequence_idx 0 \
    --video_path ./sdd_datasets/video/bookstore/video0/video.mp4 \
    --output_path ./threat_viz_track20.mp4 \
    --scale 1000.0 \
    --visualize_mode all
```

**출력**: `./threat_viz_bookstore_video0_track20_seq0.mp4`

> 💡 **팁**: 모델 학습 없이도 threat score 시각화가 가능합니다. 데이터로더에서 threat score를 자동으로 계산합니다.

---

## 데이터셋 설정

### SDD 데이터셋 구조

```
/raid/guest/SDD_2beon/sdd_bookstore/
  ├── train/
  │   ├── bookstore_video0_train.txt
  │   └── ...
  ├── val/
  │   └── ...
  └── test/
      └── bookstore_video0_test.txt
```

### 데이터 파일 형식

ETH/UCY 형식 (권장):
- 형식: `<frame_id>\t<track_id>\t<x>\t<y>`
- 구분자: 탭 (`\t`)
- 좌표: 정규화된 좌표 또는 픽셀 좌표

### 라벨 파일 (선택사항)

라벨 CSV: `track_id → class` 매핑
```csv
track_id,label
12,Pedestrian
43,Biker
```

클래스 크기 CSV: `class → size` 매핑
```csv
label,size
Pedestrian,1.0
Biker,1.2
Car,6.0
```

### 사용 예시

```bash
python train.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/train \
    --labels_dir /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore \
    --class_size_csv /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore/class_sizes.csv
```

---

## 모델 학습

### 기본 학습 명령어

```bash
python train.py \
    --dataset bookstore \
    --dataset_type sdd \
    --obs_seq_len 8 \
    --pred_seq_len 12 \
    --n_stgcn 1 \
    --n_tpcnn 4 \
    --kernel_size 3 \
    --batch_size 64 \
    --num_epochs 80 \
    --lr 1e-4 \
    --use_lrschd \
    --lr_sh_rate 32 \
    --tag dmrgcn_4rel_sdd \
    --compute_metrics
```

### 주요 파라미터

- `--dataset`: 데이터셋 이름 (bookstore, coupa, etc.)
- `--dataset_type`: `sdd` 또는 `generic`
- `--obs_seq_len`: 관찰 시퀀스 길이 (기본값: 8)
- `--pred_seq_len`: 예측 시퀀스 길이 (기본값: 12)
- `--n_stgcn`: GCN 레이어 수 (기본값: 1)
- `--n_tpcnn`: CNN 레이어 수 (기본값: 4)
- `--compute_metrics`: ADE/FDE 메트릭 계산 (validation 시)

### 체크포인트

체크포인트는 `./checkpoints/<tag>/` 디렉토리에 저장됩니다:
- `<dataset>.pth`: 각 에포크의 모델
- `<dataset>_best.pth`: 최고 성능 모델
- `args.pkl`: 학습 파라미터
- `metrics.pkl`: 학습 메트릭

---

## Threat Score 시각화

### 빠른 시작

```bash
# Bookstore video0의 track ID 20 시각화
./visualize_bookstore_video0.sh 20
```

### 시각화 모드

- `arrows`: 화살표만 표시
- `circles`: 원만 표시
- `heatmap`: 히트맵만 표시
- `all`: 모든 모드 조합 (기본값)

### 사용 예시

```bash
# 기본 시각화
python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/test \
    --target_track_id 20 \
    --video_path ./sdd_datasets/video/bookstore/video0/video.mp4 \
    --output_path ./threat_viz_track20.mp4 \
    --visualize_mode all

# 프레임으로 저장
python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/test \
    --target_track_id 20 \
    --save_frames \
    --output_dir ./threat_frames/

# 화살표만 표시
python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/test \
    --target_track_id 20 \
    --visualize_mode arrows \
    --output_path ./threat_arrows.mp4
```

### 시각화 요소

- **타겟 인물**: 노란색 원
- **다른 인물/객체**: Threat score에 따른 색상 원
  - 높은 threat: 빨간색
  - 낮은 threat: 초록색
- **화살표**: Threat 관계 표시
- **히트맵**: Threat score 오버레이
- **궤적**: 과거 이동 경로

---

## 프로젝트 구조

```
DMRGCN/
├── model/
│   ├── dmrgcn.py          # DMRGCN 모델 (4-relation 지원)
│   ├── predictor.py       # Predictor (4-relation 설정)
│   ├── gcn.py             # GCN 레이어
│   ├── loss.py            # Loss 함수
│   └── ...
├── utils/
│   ├── sdd_dataloader.py  # SDD 데이터로더 (4-relation)
│   ├── affordance.py      # Threat score 계산
│   ├── threat_visualizer.py  # Threat score 시각화
│   ├── labels.py          # 라벨 로딩
│   └── ...
├── train.py               # 학습 스크립트
├── visualize_threat.py    # 시각화 스크립트
├── visualize_bookstore_video0.sh  # Bookstore video0 시각화 스크립트
├── README.md              # 이 파일
├── README_4REL_SDD.md     # 4-relation extension 상세 문서
└── README_THREAT_VISUALIZATION.md  # Threat score 시각화 상세 문서
```

### 주요 파일 설명

- **model/predictor.py**: 4-relation 설정 및 multi-scale binning
- **utils/affordance.py**: Threat score 계산 (z_ij, T_ij)
- **utils/sdd_dataloader.py**: SDD 데이터 로딩 및 4개 인접 행렬 생성
- **utils/threat_visualizer.py**: Threat score 시각화 함수
- **visualize_threat.py**: 시각화 스크립트

---

## 상세 문서

이 프로젝트의 상세한 사용법은 다음 문서를 참조하세요:

> 📚 **전체 문서 인덱스**: [README_INDEX.md](./README_INDEX.md)에서 모든 문서를 한눈에 볼 수 있습니다.

1. **[README_4REL_SDD.md](./README_4REL_SDD.md)** 📘
   - 4-relation extension 상세 설명
   - Threat score 계산 방법
   - 그래프 구성 방법
   - 모델 구조 및 학습 방법
   - **대상**: 모델 개발자, 연구자

2. **[README_THREAT_VISUALIZATION.md](./README_THREAT_VISUALIZATION.md)** 📗
   - Threat score 시각화 전체 가이드
   - 시각화 모드 설명 (arrows, circles, heatmap, all)
   - 다양한 사용 예시
   - 문제 해결 방법
   - **대상**: 모든 사용자 (특히 시각화 사용자)

3. **[VISUALIZE_BOOKSTORE_V0.md](./VISUALIZE_BOOKSTORE_V0.md)** 📙
   - Bookstore video0 전용 시각화 가이드
   - 빠른 시작 명령어
   - 사용 가능한 track ID 확인 방법
   - 예시 명령어 모음
   - **대상**: Bookstore video0 사용자

---

## 주요 기능

### 1. 4-Relation Graph Construction

```python
# 4개의 인접 행렬 생성
A = [
    A_disp,        # Displacement relation
    A_dist,        # Distance relation
    A_pp_threat,   # PP threat relation
    A_po_threat    # PO threat relation
]
```

### 2. Threat Score 계산

```python
from utils.affordance import compute_threat_scores

# Threat score 계산
T_pp, T_po = compute_threat_scores(
    pos_t,      # 위치
    vel_t,      # 속도
    sizes,      # 크기
    ped_mask,   # 보행자 마스크
    obj_mask    # 객체 마스크
)
```

### 3. 데이터 로딩

```python
from utils import SDDTrajectoryDataset

dataset = SDDTrajectoryDataset(
    data_dir='./sdd_datasets/bookstore/train/',
    obs_len=8,
    pred_len=12,
    labels_dir='./labels/bookstore/',
    class_size_csv='./labels/class_sizes.csv'
)
```

---

## 문제 해결

### 데이터 로딩 오류

**문제**: `IndexError: too many indices for array`
- **해결**: 데이터 파일 형식 확인 (ETH/UCY 형식 또는 SDD 원본 형식)
- 데이터로더가 자동으로 형식을 감지하지만, 파일이 손상되었을 수 있음

**문제**: 데이터가 로드되지 않음
- **해결**: 데이터 디렉토리 경로 확인
- 파일이 `.txt` 확장자를 가지고 있는지 확인
- 좌표 스케일 확인 (`--scale` 파라미터 조정: 1.0 또는 1000.0)

### 시각화 오류

**문제**: Track ID를 찾을 수 없음
```
Warning: Target track ID 5 not found in sequence
Available track IDs: [1, 3, 7, 9]
```
- **해결**: 
  - 사용 가능한 track ID 확인: `python -c "from utils import SDDTrajectoryDataset; dataset = SDDTrajectoryDataset('...'); print(dataset.get_track_ids(0))"`
  - `--sequence_idx` 변경하여 다른 시퀀스 시도

**문제**: 좌표가 비디오와 맞지 않음
- **해결**: `--scale` 파라미터 조정
  - 정규화된 좌표 (0-50 범위): `--scale 1000.0`
  - 픽셀 좌표 (큰 값): `--scale 1.0`

**문제**: 비디오 파일을 찾을 수 없음
- **해결**: 원본 비디오가 없어도 시각화 가능 (검은 배경 사용)
- `--video_path` 파라미터 생략 가능

### 모델 학습 오류

**문제**: GPU 메모리 부족
- **해결**: `--batch_size` 감소 (예: 64 → 32 → 16)

**문제**: 학습이 수렴하지 않음
- **해결**: 
  - 학습률 조정: `--lr` 파라미터 변경 (예: 1e-4 → 5e-5)
  - 학습률 스케줄러 사용: `--use_lrschd` 추가

**문제**: 체크포인트를 찾을 수 없음
- **해결**: `--checkpoint_dir` 또는 `--tag` 파라미터 확인
- 체크포인트 디렉토리: `./checkpoints/<tag>/`

---

## 참고 자료

### 원본 논문

```bibtex
@article{bae2021dmrgcn,
  title={Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

### 관련 프로젝트

- [DMRGCN 원본 코드](https://github.com/InhwanBae/DMRGCN)
- [Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN)

---

## 라이선스

이 프로젝트는 원본 DMRGCN 프로젝트를 기반으로 하며, 동일한 라이선스를 따릅니다.

---

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**마지막 업데이트**: 2024년 11월
