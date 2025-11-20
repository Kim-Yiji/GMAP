# Threat Score Visualization Guide

특정 인물에 대한 threat score를 영상에 시각화하는 방법을 설명합니다.

## 개요

이 도구는 DMRGCN 모델에서 계산된 threat score를 특정 인물(track_id)에 대해 시각화합니다. 다음과 같은 시각화 방식을 제공합니다:

1. **Arrows (화살표)**: 타겟 인물에서 다른 인물/객체로의 threat 관계를 화살표로 표시
2. **Circles (원)**: 각 인물의 위치에 threat score에 따른 색상의 원을 표시
3. **Heatmap (히트맵)**: Threat score를 히트맵 오버레이로 표시
4. **All (전체)**: 위의 모든 방식을 조합

## 사용 방법

### 기본 사용법

```bash
python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --target_track_id 5 \
    --sequence_idx 0 \
    --output_path ./threat_viz.mp4 \
    --labels_dir /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore \
    --class_size_csv /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore/class_sizes.csv
```

### 주요 파라미터

- `--dataset`: 데이터셋 이름 (예: bookstore, coupa)
- `--dataset_type`: 데이터셋 타입 (sdd 또는 generic)
- `--target_track_id`: 시각화할 타겟 인물의 track ID
- `--sequence_idx`: 데이터셋에서 사용할 시퀀스 인덱스 (기본값: 0)
- `--output_path`: 출력 비디오 경로
- `--output_dir`: 프레임을 이미지로 저장할 경우 디렉토리 경로
- `--video_path`: 원본 비디오 파일 경로 (선택사항, 없으면 검은 배경)
- `--scale`: 좌표 스케일 팩터 (정규화된 SDD의 경우 1000.0)
- `--visualize_mode`: 시각화 모드 (arrows, circles, heatmap, all)
- `--save_frames`: 비디오 대신 개별 프레임으로 저장
- `--checkpoint_dir`: 모델 체크포인트 디렉토리 (선택사항)
- `--tag`: 모델 태그 (기본값: dmrgcn_4rel_sdd)

### 시각화 모드 예시

#### 1. 화살표만 표시
```bash
python visualize_threat.py \
    --dataset bookstore \
    --target_track_id 5 \
    --visualize_mode arrows \
    --output_path ./threat_arrows.mp4
```

#### 2. 원과 히트맵 조합
```bash
python visualize_threat.py \
    --dataset bookstore \
    --target_track_id 5 \
    --visualize_mode all \
    --output_path ./threat_all.mp4
```

#### 3. 프레임으로 저장
```bash
python visualize_threat.py \
    --dataset bookstore \
    --target_track_id 5 \
    --save_frames \
    --output_dir ./threat_frames/
```

#### 4. 원본 비디오 위에 오버레이
```bash
python visualize_threat.py \
    --dataset bookstore \
    --target_track_id 5 \
    --video_path /path/to/original/video.mp4 \
    --output_path ./threat_overlay.mp4
```

## Threat Score 색상 매핑

Threat score는 0-1 범위의 값이며, 다음과 같이 색상으로 매핑됩니다:

- **낮은 threat (0.0)**: 초록색 (Green)
- **중간 threat (0.5)**: 노란색 (Yellow)
- **높은 threat (1.0)**: 빨간색 (Red)

화살표와 원의 크기/두께는 threat score에 비례하여 조정됩니다.

## 사용 예시

### 1. 특정 시퀀스의 모든 track ID 확인

먼저 데이터셋에서 사용 가능한 track ID를 확인할 수 있습니다:

```python
from utils import SDDTrajectoryDataset

dataset = SDDTrajectoryDataset('./sdd_datasets/bookstore/val/')
track_ids = dataset.get_track_ids(0)  # 첫 번째 시퀀스의 track IDs
print(f"Available track IDs: {track_ids}")
```

### 2. 여러 인물에 대한 시각화

각 인물에 대해 개별적으로 시각화할 수 있습니다:

```bash
# Track ID 5에 대한 시각화
python visualize_threat.py --dataset bookstore --target_track_id 5 --output_path ./threat_track5.mp4

# Track ID 10에 대한 시각화
python visualize_threat.py --dataset bookstore --target_track_id 10 --output_path ./threat_track10.mp4
```

### 3. 여러 시퀀스 시각화

다른 시퀀스 인덱스를 사용하여 여러 시퀀스를 시각화할 수 있습니다:

```bash
python visualize_threat.py --dataset bookstore --target_track_id 5 --sequence_idx 0 --output_path ./threat_seq0.mp4
python visualize_threat.py --dataset bookstore --target_track_id 5 --sequence_idx 1 --output_path ./threat_seq1.mp4
```

## 시각화 요소 설명

### 타겟 인물
- **노란색 원**으로 표시
- 두꺼운 검은 테두리

### 다른 인물/객체
- **Threat score에 따른 색상 원**:
  - 높은 threat: 빨간색, 큰 원
  - 낮은 threat: 초록색, 작은 원
- Threat score가 0.1 미만인 경우 회색 작은 원으로 표시

### 화살표
- 타겟 인물에서 다른 인물/객체로의 threat 관계를 화살표로 표시
- 화살표 색상과 두께는 threat score에 비례

### 히트맵
- 타겟 인물 주변의 threat score를 히트맵으로 오버레이
- 투명도 30%로 원본 프레임 위에 표시

### 궤적
- 과거 궤적을 선으로 표시
- 타겟 인물: 초록색 두꺼운 선
- 다른 인물: 회색 얇은 선

## 문제 해결

### Track ID를 찾을 수 없음
```
Warning: Target track ID 5 not found in sequence
Available track IDs: [1, 3, 7, 9]
```

해결 방법: 사용 가능한 track ID 중 하나를 선택하거나, `sequence_idx`를 변경하여 다른 시퀀스를 시도하세요.

### 비디오 파일을 찾을 수 없음
원본 비디오 파일이 없어도 시각화는 가능합니다. 검은 배경 위에 궤적과 threat score만 표시됩니다.

### 좌표가 잘못 표시됨
`--scale` 파라미터를 조정하세요. 정규화된 SDD 데이터의 경우 1000.0을 사용합니다.

## 코드 구조

- `utils/threat_visualizer.py`: Threat score 시각화 핵심 함수
- `visualize_threat.py`: 시각화 스크립트
- `utils/sdd_dataloader.py`: Track IDs 저장 기능 추가

## 참고사항

1. Threat score는 데이터로더에서 계산되므로 모델이 없어도 시각화 가능합니다.
2. 원본 비디오가 있는 경우 더 나은 시각화 결과를 얻을 수 있습니다.
3. 여러 시각화 모드를 조합하여 다양한 관점에서 threat score를 분석할 수 있습니다.


