# DMRGCN 그래프 구성 방식 분석

## 개요
이 문서는 OATMeal_Queens/Threatscore의 DMRGCN 모델이 각 개체(보행자)를 어떻게 그래프로 묶는지 분석한 내용입니다.

## 1. 그래프 구성 과정 개요

DMRGCN은 보행자 궤적 예측을 위해 각 개체를 노드로 하는 그래프를 구성합니다. 전체 과정은 다음과 같습니다:

1. **데이터 로딩**: 보행자 궤적 데이터를 시퀀스로 변환
2. **그래프 변환**: 시퀀스를 그래프 구조로 변환 (`seq_to_graph`)
3. **인접 행렬 생성**: 개체 간 관계를 나타내는 인접 행렬(A) 생성
4. **다중 관계 분리**: 인접 행렬을 여러 관계 유형으로 분리
5. **그래프 컨볼루션**: 분리된 관계별로 그래프 컨볼루션 수행

## 2. 상세 분석

### 2.1 시퀀스에서 그래프로 변환 (`seq_to_graph`)

**위치**: `utils/dataloader.py`의 `seq_to_graph` 함수 (17-34줄)

```python
def seq_to_graph(seq, seq_rel):
    num_nodes = seq.shape[0]  # 개체(보행자) 수
    seq_len = seq.shape[2]    # 시퀀스 길이
    
    V = torch.zeros((seq_len, num_nodes, 2), dtype=torch.float)
    A_dist = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    A_disp = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    
    for t in range(seq_len):
        for n in range(num_nodes):
            V[t, n, :] = seq_rel[n, :, t]  # 상대 속도 벡터
            for l in range(n + 1, num_nodes):
                # 거리 기반 인접 행렬
                A_dist[t, n, l] = A_dist[t, l, n] = anorm(seq[n, :, t], seq[l, :, t])
                # 변위 기반 인접 행렬
                A_disp[t, n, l] = A_disp[t, l, n] = anorm(seq_rel[n, :, t], seq_rel[l, :, t])
    
    return V, torch.stack([A_disp, A_dist], dim=0)
```

**핵심 포인트**:
- **노드**: 각 보행자가 하나의 노드
- **엣지**: 두 보행자 간의 관계를 두 가지 방식으로 계산
  - `A_dist`: 절대 위치 기반 유클리드 거리
  - `A_disp`: 상대 속도(변위) 기반 유클리드 거리
- **대칭 행렬**: 그래프가 무방향이므로 `A[i,j] = A[j,i]`

### 2.2 데이터셋에서의 그래프 생성

**위치**: `utils/dataloader.py`의 `TrajectoryDataset.__init__` (193-212줄)

```python
# Convert Trajectories to Graphs
self.V_obs = []
self.A_obs = []
self.V_pred = []
self.A_pred = []

for ss in range(len(self.seq_start_end)):
    start, end = self.seq_start_end[ss]
    # 관찰 구간 그래프 생성
    v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :])
    self.V_obs.append(v_.clone())
    self.A_obs.append(a_.clone())
    # 예측 구간 그래프 생성
    v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :])
    self.V_pred.append(v_.clone())
    self.A_pred.append(a_.clone())
```

**핵심 포인트**:
- 각 시퀀스마다 독립적인 그래프 생성
- `seq_start_end`: 같은 시퀀스에 속한 보행자들의 인덱스 범위
- 관찰 구간(`obs`)과 예측 구간(`pred`) 각각에 대해 그래프 생성

### 2.3 다중 관계 분리 (Disentangling)

**위치**: `model/dmrgcn.py`의 `get_disentangled_adjacency_matrix` 함수 (19-32줄)

```python
def get_disentangled_adjacency_matrix(A, split=[]):
    if len(split) == 0:
        return [A]
    
    split.sort()
    split = split + [1e10]
    
    A_d = []
    for i in range(len(split) - 1):
        A_d.append(clip_adjacency_matrix(A, min=split[i], max=split[i + 1]))
    
    return torch.stack(A_d, dim=1)
```

**위치**: `model/predictor.py`의 `social_dmrgcn.__init__` (64-66줄)

```python
# Disentangling Scale Set [A_disp, A_dist]
split = [[0, 1/4, 2/4, 3/4, 1],
         [0, 1/2, 1, 2, 4]]
```

**핵심 포인트**:
- **A_disp (변위 기반)**: [0, 0.25, 0.5, 0.75, 1] 구간으로 분리 → 4개 서브그래프
- **A_dist (거리 기반)**: [0, 0.5, 1, 2, 4] 구간으로 분리 → 4개 서브그래프
- 각 구간은 서로 다른 스케일의 관계를 나타냄
  - 예: A_disp의 [0, 0.25]는 매우 작은 상대 속도 차이
  - 예: A_dist의 [2, 4]는 중간 거리 관계

### 2.4 DMRGCN의 그래프 처리

**위치**: `model/dmrgcn.py`의 `st_dmrgcn.forward` (218-236줄)

```python
def forward(self, x, A):
    res = self.residual(x)
    
    A_r = torch.split(A, 1, dim=1)  # A_disp와 A_dist로 분리
    for r in range(self.relation):  # relation=2 (A_disp, A_dist)
        # 각 관계별로 스케일 분리
        A_ = get_disentangled_adjacency_matrix(A_r[r].squeeze(dim=1), self.split[r])
        x_a, _ = self.gcns[r](x, A_)
        
        if r == 0:
            x_r = x_a
        else:
            x_r = x_r + x_a  # 관계별 결과 합산
    
    x = self.tcn(x_r) + res
    return x, A
```

**핵심 포인트**:
1. **관계 분리**: A를 A_disp와 A_dist로 분리
2. **스케일 분리**: 각 관계를 여러 스케일로 분리
3. **멀티-릴레이션 GCN**: 각 스케일별로 독립적인 GCN 적용
4. **결과 합산**: 모든 관계와 스케일의 결과를 합산

### 2.5 MultiRelationalGCN의 동작

**위치**: `model/dmrgcn.py`의 `MultiRelationalGCN.forward` (57-66줄)

```python
def forward(self, x, A):
    assert A.size(1) == self.relation  # 스케일 수
    assert A.size(2) == self.kernel_size
    
    x = self.conv(x)
    x = x.view(x.size(0), self.relation, self.out_channels, x.size(-2), x.size(-1))
    # 정규화된 라플라시안 행렬과 곱셈
    x = torch.einsum('nrtwv,nrctv->nctw', 
                     normalized_laplacian_tilde_matrix(drop_edge(A, 0.8, self.training)), 
                     x)
    return x.contiguous(), A
```

**핵심 포인트**:
- 각 스케일별 인접 행렬에 대해 독립적으로 그래프 컨볼루션 수행
- `drop_edge`: 학습 시 80% 확률로 엣지 드롭아웃 (과적합 방지)
- `normalized_laplacian_tilde_matrix`: 정규화된 라플라시안 행렬 사용

## 3. 그래프 구조 요약

### 3.1 노드
- **의미**: 각 보행자(개체)
- **특징**: 2D 좌표 (x, y) 및 상대 속도 벡터

### 3.2 엣지
- **타입 1 - A_disp (변위 기반)**:
  - 두 보행자의 상대 속도 차이
  - 작은 값: 비슷한 속도로 움직임
  - 큰 값: 서로 다른 방향/속도로 움직임
  
- **타입 2 - A_dist (거리 기반)**:
  - 두 보행자 간의 공간적 거리
  - 작은 값: 가까이 있음
  - 큰 값: 멀리 있음

### 3.3 스케일 분리
각 관계 타입은 여러 스케일로 분리되어 처리됩니다:

**A_disp 분리**:
- Scale 0: [0, 0.25) - 매우 유사한 속도
- Scale 1: [0.25, 0.5) - 유사한 속도
- Scale 2: [0.5, 0.75) - 중간 속도 차이
- Scale 3: [0.75, 1) - 큰 속도 차이

**A_dist 분리**:
- Scale 0: [0, 0.5) - 매우 가까움
- Scale 1: [0.5, 1) - 가까움
- Scale 2: [1, 2) - 중간 거리
- Scale 3: [2, 4) - 먼 거리

## 4. 데이터 흐름

```
원본 데이터 (frame_id, ped_id, x, y)
    ↓
TrajectoryDataset: 시퀀스로 변환
    ↓
seq_to_graph: 그래프 구조로 변환
    ├─ V: 노드 특성 (상대 속도)
    └─ A: 인접 행렬 [A_disp, A_dist]
        ↓
social_dmrgcn.forward
    ├─ A_disp → 스케일 분리 (4개)
    ├─ A_dist → 스케일 분리 (4개)
    ├─ 각 스케일별 MultiRelationalGCN 적용
    └─ 결과 합산
        ↓
TPCNN: 시간적 예측
    ↓
예측된 궤적
```

## 5. 주요 특징

1. **동적 그래프**: 각 시퀀스마다 보행자 수가 다르므로 그래프 크기가 가변적
2. **이중 관계**: 거리와 변위 두 가지 관계를 동시에 고려
3. **다중 스케일**: 각 관계를 여러 스케일로 분리하여 세밀한 상호작용 모델링
4. **시공간 통합**: 공간적 그래프 컨볼루션과 시간적 컨볼루션을 결합

## 6. 코드 참조 위치

- 그래프 생성: `utils/dataloader.py:17-34` (`seq_to_graph`)
- 데이터셋 처리: `utils/dataloader.py:193-212`
- 스케일 분리: `model/dmrgcn.py:19-32` (`get_disentangled_adjacency_matrix`)
- DMRGCN 모델: `model/dmrgcn.py:181-236` (`st_dmrgcn`)
- 전체 파이프라인: `model/predictor.py:58-95` (`social_dmrgcn`)

