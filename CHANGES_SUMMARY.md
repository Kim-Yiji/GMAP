# 코드 변경사항 요약

SDD bookstore 데이터셋의 `.pt` 캐시 파일을 사용하여 학습/테스트할 수 있도록 수정한 내용입니다.

## 1. `utils/dataloader.py`

### 추가된 내용
- **`CachedTrajectoryDataset` 클래스 추가** (230-274줄)
  - 전처리된 `.pt` 파일을 직접 로드하는 데이터셋 클래스
  - `TrajectoryDataset`과 동일한 인터페이스 제공
  - 캐시 파일에서 모든 필요한 텐서와 리스트를 로드

### 수정된 내용
- **108줄**: `.txt` 파일만 처리하도록 필터링 추가
  ```python
  all_files = [os.path.join(self.data_dir, _path) for _path in all_files if _path.endswith('.txt')]
  ```

## 2. `utils/__init__.py`

### 수정된 내용
- **1줄**: `CachedTrajectoryDataset` import 추가
  ```python
  from .dataloader import TrajectoryDataset, CachedTrajectoryDataset
  ```

## 3. `train.py`

### 추가된 내용
- **8줄**: `CachedTrajectoryDataset` import 추가
- **41-43줄**: 새로운 커맨드라인 인자 추가
  - `--use_cache`: 캐시 파일 사용 여부
  - `--train_cache`: train 캐시 파일 경로 (자동 감지 가능)
  - `--val_cache`: val 캐시 파일 경로 (자동 감지 가능)

### 수정된 내용
- **53-82줄**: 데이터 로딩 로직 수정
  - `--use_cache` 플래그에 따라 `CachedTrajectoryDataset` 또는 `TrajectoryDataset` 사용
  - 캐시 파일 경로 자동 감지 기능 추가

## 4. `test.py`

### 추가된 내용
- **6줄**: `CachedTrajectoryDataset` import 추가
- **18-19줄**: 새로운 커맨드라인 인자 추가
  - `--use_cache`: 캐시 파일 사용 여부
  - `--test_cache`: test 캐시 파일 경로 (자동 감지 가능)
- **54-62줄**: 모델 로딩 진행 상황 출력 추가
- **76줄**: 테스트 시작 메시지 추가
- **81-152줄**: try-except 블록으로 에러 처리 추가

### 주요 수정된 내용
1. **34-51줄**: 데이터 로딩 로직 수정
   - `--use_cache` 플래그에 따라 캐시 파일 또는 텍스트 파일 사용

2. **87-121줄**: 차원 처리 로직 수정
   - `V_pred`의 차원 순서를 `generate_statistics_matrices`에 맞게 조정
   - `generate_statistics_matrices`에 4차원 텐서 전달 (batch 차원 포함)
   - 결과에서 batch 차원 제거 후 처리

3. **114-121줄**: `view` → `reshape` 변경
   - 메모리 레이아웃 문제 해결을 위해 `contiguous()` 호출 후 `reshape` 사용

4. **123-137줄**: 절대 궤적 계산 로직 수정
   - `V_obs_traj[:, :, -1]`로 각 pedestrian의 마지막 관측 좌표 사용
   - 브로드캐스팅을 위해 차원 확장

5. **139-157줄**: ADE/FDE 계산 로직 수정
   - `V_pred_traj_gt` 차원을 `(1, pred_len, num_peds, 2)`로 변환하여 비교

## 5. `model/loss.py`

### 기존 변경사항 (이미 있었던 것)
- **51-52줄**: 디버그 출력 (pdf.shape, cov.det().shape)
- **54줄**: `squeeze()` → `squeeze(-1).squeeze(-1)` 변경
- **53줄**: `cov.det().clamp(min=1e-12)` 추가 (수치 안정성)

## 사용 방법

### 학습
```bash
python train.py --dataset sdd_bookstore --use_cache \
  --train_cache /raid/guest/SDD_2beon/sdd_bookstore/train/preproc_cache_obs8_pred12_skip1.pt \
  --val_cache /raid/guest/SDD_2beon/sdd_bookstore/val/preproc_cache_obs8_pred12_skip1.pt \
  --obs_seq_len 8 --pred_seq_len 12 --batch_size 128 --num_epochs 50 \
  --tag sdd-bookstore-cached
```

### 테스트
```bash
python test.py --tag sdd-bookstore-cached --use_cache \
  --test_cache /raid/guest/SDD_2beon/sdd_bookstore/test/preproc_cache_obs8_pred12_skip1.pt \
  --n_samples 20
```

## 주요 변경 이유

1. **성능 향상**: 전처리된 `.pt` 파일을 직접 로드하여 매번 텍스트 파일을 파싱하는 시간 절약
2. **차원 불일치 해결**: `generate_statistics_matrices` 함수가 4차원 텐서를 기대함을 확인하고 수정
3. **메모리 레이아웃 문제 해결**: `view` 대신 `reshape` 사용으로 메모리 연속성 문제 해결
4. **에러 처리 강화**: 디버깅을 위한 에러 메시지 및 스택 트레이스 추가

