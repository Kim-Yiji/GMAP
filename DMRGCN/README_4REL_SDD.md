## DMRGCN + SDD 4-Relation Threat Extension

This document describes the 4-relation, multi-scale GCN extension for the Stanford Drone Dataset (SDD) with threat-aware edges. Read this first when working on or reviewing this branch.

### Key Goals
- **4 relations** across multi-scales:
  - R_PP^disp: pedestrian–pedestrian relative displacement magnitude
  - R_PP^dist: pedestrian–pedestrian distance (meters)
  - R_PP^threat: pedestrian–pedestrian threat score
  - R_PO^threat: pedestrian–dynamic-object threat score
- **Affordance features** z_ij and **threat** T_ij ∈ [0,1]
- **DropEdge** applied uniformly across relations
- **Predictor**: GCN → TCN → GTA, input 8 frames, output 12 frames
- **Training/Eval**: supports ADE/FDE; ablations by toggling relations

---

## What Changed

### New Files
- `utils/affordance.py`
  - `compute_threat_scores(pos_t, vel_t, sizes, ped_mask, obj_mask)`
    - Builds per-frame threat matrices (PP and PO) from positions, velocities, sizes
    - z_ij = [d_ij, v^+_ij, size_j, ttc_ij]; normalized; T_ij = sigmoid(w·z)
- `utils/sdd_dataloader.py`
  - `SDDTrajectoryDataset`: reads extended SDD records and produces 4 adjacencies per frame
  - Exposes A = [A_disp, A_dist, A_PP_threat, A_PO_threat]

### Modified Files
- `model/predictor.py`
  - `relation = 4`
  - Multi-scale bins per relation:
    - R_PP^disp: `[0, 0.25, 0.5, 0.75, 1]` (normalized disp)
    - R_PP^dist: `[0, 0.5, 1, 2, 4]` (meters)
    - R_PP^threat: `[0.2, 0.4, 0.6, 0.8]`
    - R_PO^threat: `[0.2, 0.4, 0.6, 0.8]`
- `utils/__init__.py`
  - Re-exports `SDDTrajectoryDataset`
- `train.py`
  - `--dataset_type {generic,sdd}` to switch loader
  - SDD path base: `./sdd_datasets/<scene>/`
  - Optional ADE/FDE logging during validation: `--compute_metrics --kstpes N`

### Unchanged but Relevant
- `model/dmrgcn.py`
  - `st_dmrgcn` already supports per-relation multi-scale via `get_disentangled_adjacency_matrix` and applies `DropEdge` uniformly inside the GCN op.

---

## Data Assumptions (SDD)

We expect SDD files with extended fields per line:

```text
<frame_id> <track_id> <x> <y> <type> <bbox_w> <bbox_h>
```

- `<type>` is one of SDD agent types. Masks:
  - Pedestrians: `{Pedestrian}`
  - Dynamic obstacles: `{Skater, Biker, Car, Bus, Cart}`
- `<bbox_w>, <bbox_h>` are used to compute approximate area (size). If unknown, they can be 0.

Directory structure (per scene):

```text
sdd_datasets/
  bookstore/
    train/  # text files
    val/
    test/   # optional; `test.py` currently uses generic loader; adapt if needed
```

---

## Graph Construction

For each sequence window and each timestep t in the observed window:

- Nodes: all tracks in the sequence (pedestrians + selected dynamic obstacles)
- Node features V[t, n, :] = relative offset (velocity) at t
- Adjacencies A (stacked on relation axis):
  1. `A_disp[t, i, j] = ||Δp_i(t) - Δp_j(t)||`
  2. `A_dist[t, i, j] = ||p_i(t) - p_j(t)||`
  3. `A_PP_threat[t] = T_pp(pos_t, vel_t, sizes)[PP pairs only]`
  4. `A_PO_threat[t] = T_po(pos_t, vel_t, sizes)[P→O pairs]`

Threat features and score:

```text
z_ij = [
  d_ij,                          # distance
  v^+_ij,                        # approach speed (positive component)
  size_j,                        # obstacle size (bbox area)
  ttc_ij = d_ij / max(eps, v^+_ij)
]
T_ij = sigmoid(w · normalize(z_ij))
```

Masks ensure PP-threat hits pedestrian–pedestrian pairs, PO-threat hits pedestrian (row) to object (column) pairs.

Multi-scale binning is applied per relation using `get_disentangled_adjacency_matrix` inside `st_dmrgcn`.

---

## Model Flow

- `A` shape into GCN: `(N, R, T, V, V)`; here `R = 4`
- Spatial: `MultiRelationalGCN` per relation with its scale bins; sums relation outputs
- Temporal: TCN + residual + PReLU
- Head: TCN stack + GTA (as in original code) to produce sequence outputs
- DropEdge: applied inside GCN on every adjacency slice with the same probability (default p=0.2 from existing code path)

---

## Training

Example command (SDD bookstore):

```bash
python DMRGCN/train.py \
  --dataset bookstore \
  --dataset_type sdd \
  --obs_seq_len 8 --pred_seq_len 12 \
  --n_stgcn 1 --n_tpcnn 4 --kernel_size 3 \
  --batch_size 64 --num_epochs 80 \
  --lr 1e-4 --use_lrschd --lr_sh_rate 32 \
  --tag dmrgcn_4rel_sdd \
  --compute_metrics --kstpes 20
```

Notes:
- Checkpoints and TensorBoard logs under `./checkpoints/<tag>/`.
- For non-SDD datasets, keep `--dataset_type generic` (default). This uses the original 2-relation loader.

---

## Evaluation

- `test.py` currently uses the generic dataset loader. For SDD, either:
  - Adapt `test.py` to use `SDDTrajectoryDataset`, or
  - Evaluate during validation using `--compute_metrics` (ADE/FDE logged each epoch).

Sampling-based ADE/FDE follows the existing multivariate Normal sampling in this repo.

---

## Ablation Experiments

To ablate threat relations:
- Easiest path: zero out the corresponding adjacency channels in the SDD loader or before passing to the model.
  - Remove `R_PP^threat`: set `A[2] = 0`
  - Remove `R_PO^threat`: set `A[3] = 0`

Alternatively, adjust `relation` and `split` in `model/predictor.py` and update loader to emit matching channels.

---

## Design Choices & Tunables

- Threat weights `w` (default `[1.0, 1.0, 0.5, 1.0]`): tweak in `utils/affordance.py`
- Threat bins: adjust in `model/predictor.py`
- Normalization ranges in `normalize_features`: tune per scene if needed
- DropEdge probability: follows existing implementation (default 0.2). Change in graph op if necessary

---

## Limitations / Future Work

- The threat function is fixed (sigmoid of weighted sum). You can replace with a learnable MLP for end-to-end training.
- `test.py` is not yet SDD-aware; adapt if standalone SDD testing is preferred.
- If SDD files do not include type/box sizes, PO-threat and size_j will degrade; consider estimating sizes or setting dynamic-object types via external metadata.

---

## Quick Reference

- Use SDD loader: `--dataset_type sdd` and scene directory under `./sdd_datasets/<scene>/`
- 4 relations order in adjacency tensor: `[disp, dist, pp_threat, po_threat]`
- Multi-scale bins are per-relation; configured in `model/predictor.py`
- Threat construction is in `utils/affordance.py`

---

## File Map (Touched)

- `model/predictor.py` — relation=4, per-relation splits
- `utils/affordance.py` — threat feature + score
- `utils/sdd_dataloader.py` — SDD loader building 4 adjacencies
- `utils/__init__.py` — export new loader
- `train.py` — SDD switch, ADE/FDE logging option

---

## ETH/UCY 형식 + 라벨 CSV(영상별) 워크플로우

ETH/UCY 전처리 형식(한 줄: `<frame> <track_id> <x> <y>`)을 사용하고, 별도의 라벨 CSV에서 `pid -> class`를 제공하며, 클래스별 size는 별도 CSV로 제공합니다.

- 라벨 CSV 위치 예: `/raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/<scene>/<video>_labels.csv`
  - 헤더: `track_id,label`
- 클래스 사이즈 CSV 예: `<labels_dir>/class_sizes.csv`
  - 헤더: `label,size`

사용 예시:

```bash
python DMRGCN/train.py \
  --dataset bookstore \
  --dataset_type sdd \
  --labels_dir /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore \
  --class_size_csv /raid/guest/OATMeal_Queens/SDD_datasets/SDD_labels/bookstore/class_sizes.csv \
  --obs_seq_len 8 --pred_seq_len 12 \
  --n_stgcn 1 --n_tpcnn 4 --kernel_size 3 \
  --batch_size 64 --num_epochs 80 \
  --lr 1e-4 --use_lrschd --lr_sh_rate 32 \
  --tag dmrgcn_4rel_sdd_labels \
  --compute_metrics --kstpes 20
```

동작 방식:
- 데이터 파일 이름의 베이스(`<video>`)로 `<video>_labels.csv`를 찾아 `track_id -> class` 매핑을 구성합니다.
- `class_sizes.csv`에서 `class -> size`를 읽어 `size_j`를 설정합니다.
- `Pedestrian`은 보행자 마스크, `{Skater,Biker,Car,Bus,Cart}`는 동적 장애물 마스크에 반영됩니다.
- bbox가 없어도 위협 relation(R_PP^threat, R_PO^threat)을 계산할 수 있습니다.

참고:
- ETH/UCY 형식 데이터 루트(예: `/raid/guest/SDD_2beon/<scene>/<split>/`)를 `--dataset bookstore --dataset_type sdd`의 기본 경로 규칙(`./sdd_datasets/<scene>/`)에 맞춰 배치하거나, 필요 시 심볼릭 링크를 사용하세요.


