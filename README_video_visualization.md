# DMRGCN 동영상 확률맵 시각화

이 스크립트는 DMRGCN 모델의 예측 결과를 동영상 위에 확률맵으로 시각화합니다.

## 기능

- DMRGCN 모델의 예측 궤적을 확률맵으로 시각화
- 실제 동영상 데이터 위에 확률맵 오버레이
- 관찰된 궤적(파란색)과 실제 궤적(빨간색) 표시
- 개별 프레임 이미지와 동영상 파일 생성

## 사용법

```bash
python3 test_with_video_visualization.py --tag <모델_태그> --dataset <데이터셋명> --frame_start <시작_프레임> --frame_end <끝_프레임>
```

### 매개변수

- `--tag`: 체크포인트 디렉토리 이름 (기본값: social-dmrgcn-hotel-experiment_tp4_de80)
- `--dataset`: 데이터셋 이름 (기본값: hotel)
- `--frame_start`: 시작 프레임 번호 (기본값: 0)
- `--frame_end`: 끝 프레임 번호 (기본값: 10)
- `--n_samples`: 확률맵 생성을 위한 샘플 수 (기본값: 1000)
- `--output_dir`: 출력 디렉토리 (기본값: ./video_visualization_output/)

### 예시

```bash
# Hotel 데이터셋으로 0-20 프레임 시각화
python3 test_with_video_visualization.py --tag social-dmrgcn-hotel-experiment_tp4_de80 --dataset hotel --frame_start 0 --frame_end 20

# 샘플 수를 줄여서 빠르게 실행
python3 test_with_video_visualization.py --tag social-dmrgcn-hotel-experiment_tp4_de80 --dataset hotel --frame_start 0 --frame_end 10 --n_samples 500
```

## 출력 파일

- `{dataset}_probability_map.avi`: 확률맵이 오버레이된 동영상 파일
- `frame_XXXX.png`: 개별 프레임 이미지 파일들

## 요구사항

- PyTorch (CPU 버전)
- OpenCV
- Matplotlib
- NumPy
- SciPy
- tqdm

## 데이터 구조

스크립트는 다음 디렉토리 구조를 가정합니다:

```
DMRGCN/
├── checkpoints/
│   └── {tag}/
│       ├── args.pkl
│       └── {dataset}_best.pth
├── datasets/
│   └── {dataset}/
│       └── test/
├── ETH-UCY-Trajectory-Visualizer/
│   └── datasets_visualize/
│       └── {dataset}/
│           ├── H.txt
│           ├── video.avi
│           └── test/
```

## 시각화 설명

- **확률맵**: 모델이 예측한 궤적의 불확실성을 색상으로 표현
  - 검은색: 낮은 확률
  - 파란색 → 청록색 → 노란색 → 빨간색: 높은 확률
- **파란색 선**: 관찰된 궤적 (과거 8프레임)
- **빨간색 선**: 실제 궤적 (미래 12프레임)

## 주의사항

- CUDA가 없는 환경에서 CPU로 실행됩니다
- 대용량 샘플 수는 처리 시간이 오래 걸릴 수 있습니다
- 동영상 파일 크기는 프레임 수와 해상도에 따라 달라집니다
