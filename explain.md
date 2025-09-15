# DMRGCN + GP-Graph 통합 모델 상세 설명

## 📋 프로젝트 개요

이 프로젝트는 두 개의 최신 보행자 궤적 예측 모델을 통합하여 더 강력한 예측 성능을 달성하는 것을 목표로 합니다:

- **DMRGCN (AAAI 2021)**: Multi-Relational Graph Convolution 기반
- **GP-Graph (ECCV 2022)**: Group-based Processing 기반

### 🎯 주요 특징

- **모듈화된 설계**: GP-Graph를 모듈로 분리하여 DMRGCN에 통합
- **추가 Feature**: Local Density, Group Size 등 새로운 특징 추가
- **모션 분석**: 다양한 보행자 모션 타입 분석 및 필터링
- **성능 비교**: 전체 데이터셋 vs 필터링된 데이터셋 성능 비교

## 🔬 각 논문의 핵심 특징과 적용 방식

### 1. DMRGCN (AAAI 2021) - "Disentangled Multi-Relational Graph Convolutional Network"

#### 📖 논문의 핵심 아이디어
DMRGCN은 보행자 궤적 예측에서 **사회적 상호작용의 복잡성**을 해결하기 위해 설계되었습니다.

**주요 문제점들:**
- **Over-smoothing 문제**: 고차원 사회적 관계에서 발생하는 과도한 평활화
- **Biased weighting 문제**: 높은 차수의 사회적 관계에서 발생하는 편향된 가중치
- **Accumulated errors**: 보행자가 방향을 바꿀 때 누적되는 오차

#### 🧠 핵심 기술적 특징

**1. Disentangled Multi-scale Aggregation (분리된 다중 스케일 집계)**
```python
# model/dmrgcn.py의 st_dmrgcn 클래스에서 구현
class st_dmrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, split=[], relation=2):
        # Spatial Edge - 다중 관계형 GCN
        self.gcns = nn.ModuleList()
        for r in range(self.relation):
            self.gcns.append(MultiRelationalGCN(in_channels, out_channels, kernel_size[1], 
                                              relation=(len(split[r]))))
```

**근거**: `get_disentangled_adjacency_matrix` 함수에서 인접 행렬을 여러 스케일로 분리
```python
def get_disentangled_adjacency_matrix(A, split=[]):
    # split = [[0, 1/4, 2/4, 3/4, 1], [0, 1/2, 1, 2, 4]]
    # 거리 기반으로 인접 행렬을 여러 구간으로 분리
    for i in range(len(split) - 1):
        A_d.append(clip_adjacency_matrix(A, min=split[i], max=split[i + 1]))
    return torch.stack(A_d, dim=1)
```

**2. Global Temporal Aggregation (전역 시간적 집계)**
```python
# model/predictor.py의 tpcnn 클래스에서 구현
class tpcnn(nn.Module):
    def __init__(self, seq_len, pred_seq_len, output_feat, n_tpcn=2, n_gtacn=1):
        # Global Temporal Aggregation (GTA)
        self.gtacn = nn.ModuleList()
        self.gtacn.append(nn.Sequential(
            nn.Conv2d(output_feat, output_feat, (pred_seq_len, 1), padding=0),
            nn.PReLU(),
            nn.Dropout(dropout, inplace=True)
        ))
```

**근거**: `(pred_seq_len, 1)` 커널을 사용하여 전체 예측 시퀀스에 걸쳐 전역적 집계 수행

**3. DropEdge Technique (드롭 엣지 기법)**
```python
# model/dmrgcn.py의 MultiRelationalGCN에서 구현
x = torch.einsum('nrtwv,nrctv->nctw', 
                 normalized_laplacian_tilde_matrix(drop_edge(A, 0.8, self.training)), x)
```

**근거**: `drop_edge` 함수에서 80% 확률로 엣지를 제거하여 과적합 방지

#### 🔧 우리 프로젝트에서의 적용

**1. 베이스 모델로 사용**
```python
# model/social_dmrgcn_gpgraph.py에서
self.dmrgcn = social_dmrgcn(
    n_stgcn=n_stgcn, n_tpcnn=n_tpcnn, input_feat=2, output_feat=output_feat,
    kernel_size=kernel_size, seq_len=seq_len, pred_seq_len=pred_seq_len
)
```

**2. Multi-Relational Processing 유지**
- 원본 DMRGCN의 `st_dmrgcn` 클래스 그대로 사용
- `split=[[0, 1/4, 2/4, 3/4, 1], [0, 1/2, 1, 2, 4]]` 설정으로 거리 기반 분리 유지
- `relation=2`로 공간적 관계와 시간적 관계 분리

**3. 속도, 가속도, 상대변위 특징 제공**
```python
# utils/dataloader.py에서
def seq_to_graph(seq, seq_rel):
    # 상대 변위 계산
    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
    
    # 거리와 변위 기반 인접 행렬 생성
    A_dist[t, n, l] = A_dist[t, l, n] = anorm(seq[n, :, t], seq[l, :, t])
    A_disp[t, n, l] = A_disp[t, l, n] = anorm(seq_rel[n, :, t], seq_rel[l, :, t])
    
    return V, torch.stack([A_disp, A_dist], dim=0)
```

### 2. GP-Graph (ECCV 2022) - "Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction"

#### 📖 논문의 핵심 아이디어
GP-Graph는 보행자들의 **그룹 행동 패턴**을 학습하여 더 정확한 궤적 예측을 수행합니다.

**주요 문제점들:**
- **Individual-level prediction의 한계**: 개별 보행자만 고려한 예측의 한계
- **Group dynamics 무시**: 보행자 그룹의 집단적 행동 패턴 미고려
- **Multi-modal prediction**: 다양한 가능한 미래 궤적 예측의 어려움

#### 🧠 핵심 기술적 특징

**1. Unsupervised Group Estimation (비지도 그룹 추정)**
```python
# model/gpgraph_modules.py의 GroupGenerator 클래스에서 구현
class GroupGenerator(nn.Module):
    def forward(self, v, v_abs, tau=0.1, hard=True):
        # 보행자 간 유사도 계산
        if self.d_type == 'learned_l2norm':
            temp = self.group_cnn(v_abs).unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        
        # 그룹 인덱스 찾기
        indices = self.find_group_indices(v, dist_mat)
        return v, indices
```

**근거**: `find_group_indices` 함수에서 거리 임계값을 기반으로 그룹 형성
```python
def find_group_indices(self, v, dist_mat):
    # 거리 임계값 이하의 보행자들을 같은 그룹으로 할당
    top_row, top_column = torch.nonzero(dist_mat.tril(diagonal=-1).add(mask).le(self.th), as_tuple=True)
    # Union-Find 알고리즘으로 그룹 통합
    for r, c in zip(top_row, top_column):
        mask = indices_raw == indices_raw[r]
        indices_raw[mask] = c
```

**2. Pedestrian Group Pooling/Unpooling (보행자 그룹 풀링/언풀링)**
```python
@staticmethod
def ped_group_pool(v, indices):
    # 그룹별로 특징을 평균하여 풀링
    v_pool = torch.zeros(v.shape[:-1] + (n_ped_pool,), device=v.device)
    v_pool.index_add_(-1, indices, v)
    v_pool_num = torch.zeros((v.size(0), 1, 1, n_ped_pool), device=v.device)
    v_pool_num.index_add_(-1, indices, torch.ones((v.size(0), 1, 1, n_ped), device=v.device))
    v_pool /= v_pool_num  # 평균 계산
    return v_pool

@staticmethod
def ped_group_unpool(v, indices):
    # 그룹 특징을 개별 보행자로 다시 분배
    return torch.index_select(input=v, dim=-1, index=indices)
```

**근거**: 그룹 레벨에서 처리한 특징을 다시 개별 보행자 레벨로 복원

**3. Group Hierarchy Graph (그룹 계층 그래프)**
```python
# model/social_dmrgcn_gpgraph.py에서 구현
# Inter-group processing (그룹 간 상호작용)
if self.group_type[1]:
    v_e = self.group_gen.ped_group_pool(v_rel, group_indices)
    A_e = generate_adjacency_matrix(v_e)
    v_e_pred, _ = self.dmrgcn(v_e, A_e.unsqueeze(0).unsqueeze(0))
    v_e_pred = self.group_gen.ped_group_unpool(v_e_pred, group_indices)

# Intra-group processing (그룹 내 상호작용)
if self.group_type[2]:
    mask = self.group_gen.ped_group_mask(group_indices)
    A_i = generate_adjacency_matrix(v_i) * mask
    v_i_pred, _ = self.dmrgcn(v_i, A_i.unsqueeze(0).unsqueeze(0))
```

**근거**: `ped_group_mask` 함수에서 그룹 내 연결만 허용하는 마스크 생성
```python
@staticmethod
def ped_group_mask(indices):
    mask = torch.eye(indices.size(0), dtype=torch.bool, device=indices.device)
    for i in indices.unique():
        idx_list = torch.nonzero(indices.eq(i))
        for idx in idx_list:
            mask[idx, idx_list] = 1  # 같은 그룹 내에서만 연결
    return mask
```

#### 🔧 우리 프로젝트에서의 적용

**1. 모듈화된 설계**
```python
# model/gpgraph_modules.py에서 GP-Graph 컴포넌트들을 독립 모듈로 분리
class GroupGenerator(nn.Module):  # 그룹 생성
class GroupIntegrator(nn.Module):  # 그룹 통합
class DensityGroupFeatureExtractor(nn.Module):  # 추가 특징 추출
```

**2. DMRGCN과의 통합**
```python
# model/social_dmrgcn_gpgraph.py에서
if self.use_group_processing:
    # 1. Original individual-level processing
    if self.group_type[0]:
        v_orig = v_pred_base
        v_stack.append(v_orig)
    
    # 2. Inter-group processing
    if self.group_type[1]:
        v_e = self.group_gen.ped_group_pool(v_rel, group_indices)
        v_e_pred, _ = self.dmrgcn(v_e, A_e.unsqueeze(0).unsqueeze(0))
        v_e_pred = self.group_gen.ped_group_unpool(v_e_pred, group_indices)
        v_stack.append(v_e_pred)
    
    # 3. Intra-group processing
    if self.group_type[2]:
        v_i_pred, _ = self.dmrgcn(v_i, A_i.unsqueeze(0).unsqueeze(0))
        v_stack.append(v_i_pred)
    
    # 4. Group integration
    v_pred_group = self.group_mix(v_stack)
```

**3. Multi-scale Processing**
- **Individual Level**: DMRGCN의 원본 처리
- **Inter-group Level**: 그룹 간 상호작용
- **Intra-group Level**: 그룹 내 상호작용
- **Integration**: 모든 레벨의 결과를 통합

### 3. 추가된 새로운 특징들 (미팅 요청사항)

#### 🧠 Local Density (지역 밀도)

**미팅에서의 요청**: "밀도local density, 그룹 크기group size 써보는 게 좋을듯!"

**구현 방식**:
```python
# model/gpgraph_modules.py의 DensityGroupFeatureExtractor에서
def compute_local_density(self, positions, radius=None):
    for b in range(batch):
        for t in range(seq_len):
            pos = positions[b, t]  # (num_ped, 2)
            for i in range(num_ped):
                # 각 보행자 주변 반경 내의 다른 보행자 수 계산
                distances = torch.norm(pos - pos[i].unsqueeze(0), dim=1)
                density[b, t, i] = (distances < radius).sum().float() - 1
```

**근거**: `density_radius=2.0` (기본값) 반경 내의 보행자 수를 계산하여 지역 밀도 측정

#### 🧠 Group Size (그룹 크기)

**구현 방식**:
```python
def compute_group_size(self, group_indices):
    unique_groups, counts = torch.unique(group_indices, return_counts=True)
    group_sizes = torch.zeros_like(group_indices, dtype=torch.float)
    
    for group_id, size in zip(unique_groups, counts):
        mask = group_indices == group_id
        group_sizes[mask] = size.float()
    
    return group_sizes
```

**근거**: `torch.unique`를 사용하여 각 그룹의 크기를 계산하고, 같은 그룹에 속한 모든 보행자에게 동일한 그룹 크기 할당

#### 🧠 속도 벡터 사용

**미팅에서의 요청**: "몸 방향은 그냥 속도 벡터로 사용하는 게 좋을듯?"

**구현 방식**:
```python
# utils/dataloader.py에서
rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
```

**근거**: 절대 좌표의 차분을 계산하여 속도 벡터를 얻고, 이를 통해 방향 정보를 간접적으로 표현

## 🔄 통합 아키텍처의 동작 원리

### 1. 전체 처리 흐름

```
Input Trajectory (v_obs, A_obs)
    ↓
1. DMRGCN Base Processing
   - Multi-Relational Graph Convolution
   - Disentangled Multi-scale Aggregation
   - Global Temporal Aggregation
    ↓
2. GP-Graph Group Processing (if enabled)
   - Group Generation (GroupGenerator)
   - Inter-group Processing (DMRGCN on groups)
   - Intra-group Processing (DMRGCN with group mask)
   - Group Integration (GroupIntegrator)
    ↓
3. Additional Features (if enabled)
   - Local Density Computation
   - Group Size Computation
    ↓
4. Feature Integration
   - Concatenate all features
   - Apply integration layers
    ↓
Output Prediction (v_pred, group_indices)
```

### 2. 특징 통합 방식

```python
# model/social_dmrgcn_gpgraph.py에서
def _build_feature_integration(self, output_feat, use_density, use_group_size, use_group_processing):
    input_dim = output_feat  # DMRGCN 기본 출력
    
    if use_density:
        input_dim += 1  # 밀도 특징 추가
    if use_group_size:
        input_dim += 1  # 그룹 크기 특징 추가
    if use_group_processing:
        input_dim += output_feat  # 그룹 처리 출력 추가
        
    if input_dim > output_feat:
        return nn.Sequential(
            nn.Conv2d(input_dim, output_feat, kernel_size=1),  # 차원 축소
            nn.ReLU(),
            nn.Conv2d(output_feat, output_feat, kernel_size=1)  # 최종 출력
        )
    else:
        return nn.Identity()
```

**근거**: 모든 특징을 concatenate한 후 1x1 convolution으로 차원을 맞춰 최종 출력 생성

### 3. 모션 분석 및 필터링

#### 🧠 모션 타입 분류

**구현 방식**:
```python
# analyze_motions.py에서
def classify_motion_type(features):
    # Stationary motion
    if features['avg_speed'] < 0.1:
        return 'stationary'
    
    # Linear motion
    if features['linearity'] > 0.9 and features['direction_change_rate'] < 0.3:
        return 'linear'
    
    # Curved motion
    if features['linearity'] < 0.7:
        return 'curved'
    
    # Direction change motion
    if features['direction_change_rate'] > 0.5 or features['max_direction_change'] > 1.0:
        return 'direction_change'
    
    # Group motion
    if features['avg_speed'] > 0.5 and features['direction_change_rate'] < 0.2:
        return 'group_motion'
```

**근거**: 
- **Linearity**: R-squared 값으로 궤적의 직선성 측정
- **Speed**: 속도 벡터의 크기로 움직임 정도 측정
- **Direction Change**: 연속된 속도 벡터 간의 각도 변화로 방향 변화 측정

#### 🧠 특징 추출

```python
def analyze_single_trajectory(traj, traj_rel):
    # 1. Linear vs Curved motion
    t = np.arange(traj.shape[1])
    poly_x = np.polyfit(t, traj[0, :], 2)  # 2차 다항식 피팅
    poly_y = np.polyfit(t, traj[1, :], 2)
    
    # R-squared 계산
    x_pred = np.polyval(poly_x, t)
    y_pred = np.polyval(poly_y, t)
    r2_x = 1 - (ss_res_x / ss_tot_x) if ss_tot_x != 0 else 0
    r2_y = 1 - (ss_res_y / ss_tot_y) if ss_tot_y != 0 else 0
    features['linearity'] = (r2_x + r2_y) / 2
    
    # 2. Speed analysis
    speeds = np.sqrt(traj_rel[0, :] ** 2 + traj_rel[1, :] ** 2)
    features['avg_speed'] = np.mean(speeds)
    features['max_speed'] = np.max(speeds)
    features['speed_variance'] = np.var(speeds)
    
    # 3. Direction changes
    directions = np.arctan2(traj_rel[1, :], traj_rel[0, :])
    direction_changes = np.abs(np.diff(directions))
    direction_changes = np.minimum(direction_changes, 2*np.pi - direction_changes)
    features['direction_change_rate'] = np.mean(direction_changes)
```

**근거**: 
- **Polynomial fitting**: 2차 다항식으로 궤적을 피팅하여 선형성 측정
- **Speed calculation**: 상대 변위의 L2 norm으로 속도 계산
- **Direction change**: 연속된 속도 벡터 간의 각도 차이로 방향 변화 측정

## 🎯 성능 평가 및 비교

### 1. 평가 지표

**ADE (Average Displacement Error)**:
```python
# compare_performance.py에서
def calculate_ade_fde(pred_traj, gt_traj):
    displacement_errors = np.sqrt(np.sum((pred - gt) ** 2, axis=0))
    ade = np.mean(displacement_errors)  # 모든 시간 단계의 평균 오차
    fde = displacement_errors[-1]  # 최종 시간 단계의 오차
```

**근거**: 궤적 예측에서 가장 널리 사용되는 평가 지표로, 예측된 궤적과 실제 궤적 간의 거리 오차를 측정

### 2. 성능 비교 방식

**전체 데이터셋 vs 필터링된 데이터셋**:
```python
# compare_performance.py에서
dataset_configs = {
    'Original Train': f'./datasets/{args.dataset}/train/',
    'Original Val': f'./datasets/{args.dataset}/val/',
    'Original Test': f'./datasets/{args.dataset}/test/',
}

if args.compare_filtered:
    filtered_datasets = [
        'linear_only', 'curved_only', 'direction_change_only',
        'group_motion_only', 'linear_curved', 'all_motions'
    ]
    
    for filtered_name in filtered_datasets:
        filtered_path = os.path.join(args.filtered_base_path, filtered_name)
        if os.path.exists(filtered_path):
            dataset_configs[f'Filtered {filtered_name}'] = filtered_path
```

**근거**: 미팅에서 요청된 "전체 데이터셋 & 원하는 모션만 추린 데이터셋 두 개로 돌려보고 성능 비교해보기"를 구현

## 🔧 모델 파라미터 상세 설명

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
