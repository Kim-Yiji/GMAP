# Bookstore Video0 Threat Score 시각화 가이드

## 빠른 시작

### 1. 기본 사용법 (스크립트 사용)

```bash
cd /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN

# Track ID 20으로 시각화
./visualize_bookstore_video0.sh 20

# 다른 Track ID로 시각화
./visualize_bookstore_video0.sh 5

# 다른 시퀀스 인덱스 사용
./visualize_bookstore_video0.sh 20 1
```

### 2. 직접 Python 명령어 사용

```bash
cd /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN

python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/test \
    --target_track_id 20 \
    --sequence_idx 0 \
    --video_path /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN/sdd_datasets/video/bookstore/video0/video.mp4 \
    --output_path ./threat_viz_bookstore_video0_track20.mp4 \
    --scale 1000.0 \
    --visualize_mode all \
    --fps 10.0 \
    --device cuda:0
```

## 사용 가능한 Track IDs

Bookstore video0에는 총 **237개의 track ID**가 있습니다.
- Track IDs: 0 ~ 236
- 예시: 0, 1, 2, 3, 4, 5, ... 20, 28, 31, 57, 65, ...

## 파라미터 설명

- `--target_track_id`: 시각화할 타겟 인물의 track ID (필수)
- `--sequence_idx`: 데이터셋에서 사용할 시퀀스 인덱스 (기본값: 0)
- `--scale`: 좌표 스케일 팩터 (기본값: 1000.0, 정규화된 좌표 사용)
- `--visualize_mode`: 시각화 모드
  - `arrows`: 화살표만
  - `circles`: 원만
  - `heatmap`: 히트맵만
  - `all`: 모든 모드 (기본값)
- `--fps`: 출력 비디오 프레임레이트 (기본값: 10.0)
- `--save_frames`: 비디오 대신 개별 프레임으로 저장

## 출력 파일

- 비디오: `./threat_viz_bookstore_video0_track{ID}_seq{IDX}.mp4`
- 프레임: `./threat_frames/` (--save_frames 사용 시)

## 예시 명령어

### Track ID 20 시각화
```bash
./visualize_bookstore_video0.sh 20
```

### Track ID 5 시각화 (프레임으로 저장)
```bash
python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/test \
    --target_track_id 5 \
    --sequence_idx 0 \
    --video_path /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN/sdd_datasets/video/bookstore/video0/video.mp4 \
    --output_dir ./threat_frames_track5 \
    --scale 1000.0 \
    --visualize_mode all \
    --save_frames \
    --device cuda:0
```

### 화살표만 표시
```bash
python visualize_threat.py \
    --dataset bookstore \
    --dataset_type sdd \
    --data_dir /raid/guest/SDD_2beon/sdd_bookstore/test \
    --target_track_id 20 \
    --video_path /raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN/sdd_datasets/video/bookstore/video0/video.mp4 \
    --output_path ./threat_arrows_track20.mp4 \
    --scale 1000.0 \
    --visualize_mode arrows \
    --device cuda:0
```

## 문제 해결

### 좌표가 맞지 않는 경우
- `--scale` 값을 조정해보세요 (1.0 또는 1000.0)
- 비디오 해상도와 좌표 시스템이 맞는지 확인

### Track ID를 찾을 수 없는 경우
- 시퀀스 인덱스를 변경해보세요 (`--sequence_idx`)
- 데이터 파일에 해당 track ID가 있는지 확인

### 비디오 파일을 찾을 수 없는 경우
- `--video_path`를 생략하면 검은 배경으로 시각화됩니다
- 원본 비디오가 없어도 threat score는 시각화 가능합니다

## 참고

- 데이터 디렉토리: `/raid/guest/SDD_2beon/sdd_bookstore/test`
- 비디오 파일: `/raid/guest/OATMeal_Queens/newbie/yoonhee/modeling/DMRGCN/sdd_datasets/video/bookstore/video0/video.mp4`
- 라벨 디렉토리: `/raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore` (선택사항)


