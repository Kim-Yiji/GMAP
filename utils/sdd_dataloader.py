import glob
import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataloader import seq_to_graph
from .threat_score import get_obstacle_size


def load_sdd_annotations(annotations_file: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    SDD annotations.csv 를 DataFrame 으로 읽고, track_id -> label 매핑을 함께 반환.
    """
    df = pd.read_csv(annotations_file)
    # 바운딩 박스 중심 좌표 계산
    df["x"] = (df["xmin"] + df["xmax"]) / 2.0
    df["y"] = (df["ymin"] + df["ymax"]) / 2.0
    # lost / occluded 프레임 제거
    df = df[(df["lost"] == 0) & (df["occluded"] == 0)]

    labels: Dict[int, str] = {}
    for track_id in df["track_id"].unique():
        track_data = df[df["track_id"] == track_id]
        if len(track_data) < 2:
            continue
        labels[int(track_id)] = str(track_data["label"].iloc[0])

    return df, labels


class SDDTrajectoryDataset(Dataset):
    """
    SDD annotations.csv 를 이용해 DMRGCN 스타일의 다중 시퀀스를 생성하는 Dataset.

    - 입력: 하나의 video 디렉토리 (예: .../SDD_datasets/SDD_raw/bookstore/video0)
    - 여러 프레임 윈도우(길이 obs_len+pred_len)를 슬라이딩하며 시퀀스를 만든다.
    - 각 시퀀스에 대해 기존 TrajectoryDataset 과 동일한 out 포맷을 반환한다.
    """

    def __init__(self, data_dir: str, obs_len: int = 8, pred_len: int = 12, skip: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        annotation_files = self._collect_annotation_files(self.data_dir)

        if not annotation_files:
            raise FileNotFoundError(f"No annotations found under {self.data_dir}")

        # 시퀀스별 저장 리스트
        self.obs_traj_list: List[torch.Tensor] = []
        self.pred_traj_list: List[torch.Tensor] = []
        self.obs_traj_rel_list: List[torch.Tensor] = []
        self.pred_traj_rel_list: List[torch.Tensor] = []
        self.non_linear_ped_list: List[torch.Tensor] = []
        self.loss_mask_list: List[torch.Tensor] = []
        self.V_obs_list: List[torch.Tensor] = []
        self.A_obs_list: List[torch.Tensor] = []
        self.V_pred_list: List[torch.Tensor] = []
        self.A_pred_list: List[torch.Tensor] = []

        total_sequences_before = 0
        for annotations_file in annotation_files:
            sequences_before_file = len(self.obs_traj_list)
            self._process_annotations_file(annotations_file)
            sequences_after_file = len(self.obs_traj_list)
            print(
                f"[SDDTrajectoryDataset] {os.path.basename(annotations_file)} → "
                f"{sequences_after_file - sequences_before_file} sequences"
            )
            total_sequences_before = sequences_after_file

        self.num_seq = len(self.obs_traj_list)
        print(f"[SDDTrajectoryDataset] Built {self.num_seq} sequences from {len(annotation_files)} file(s)")

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index: int):
        obs_traj = self.obs_traj_list[index]
        pred_traj = self.pred_traj_list[index]
        obs_traj_rel = self.obs_traj_rel_list[index]
        pred_traj_rel = self.pred_traj_rel_list[index]
        non_linear_ped = self.non_linear_ped_list[index]
        loss_mask = self.loss_mask_list[index]
        V_obs = self.V_obs_list[index]
        A_obs = self.A_obs_list[index]
        V_pred = self.V_pred_list[index]
        A_pred = self.A_pred_list[index]

        out = [
            obs_traj,
            pred_traj,
            obs_traj_rel,
            pred_traj_rel,
            non_linear_ped,
            loss_mask,
            V_obs,
            A_obs,
            V_pred,
            A_pred,
        ]
        return out

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _collect_annotation_files(self, data_dir: str) -> List[str]:
        """Collect annotation CSV files from the provided directory."""
        if os.path.isfile(data_dir):
            return [data_dir]

        annotation_files: List[str] = []

        if os.path.isdir(data_dir):
            single_file = os.path.join(data_dir, "annotations.csv")
            if os.path.isfile(single_file):
                annotation_files.append(single_file)
            else:
                annotation_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

        return annotation_files

    def _process_annotations_file(self, annotations_file: str) -> None:
        """Process a single annotations.csv file and append sequences."""
        print(f"[SDDTrajectoryDataset] Loading SDD annotations from {annotations_file}")
        df, labels = load_sdd_annotations(annotations_file)

        # 트랙별로 프레임/좌표 정리
        trajectories: Dict[int, List[Tuple[int, float, float]]] = {}
        for track_id in df["track_id"].unique():
            track_data = df[df["track_id"] == track_id].sort_values("frame")
            if len(track_data) < self.seq_len:
                continue
            traj = [(int(row["frame"]), float(row["x"]), float(row["y"])) for _, row in track_data.iterrows()]
            trajectories[int(track_id)] = traj

        if len(trajectories) == 0:
            print(f"[SDDTrajectoryDataset] No valid trajectories (len >= {self.seq_len}) in {annotations_file}")
            return

        all_frames = sorted(df["frame"].unique().tolist())
        if len(all_frames) < self.seq_len:
            print(f"[SDDTrajectoryDataset] Not enough frames in {annotations_file} to build sequences")
            return

        total_windows = max(1, (len(all_frames) - self.seq_len) // self.skip + 1)
        pbar = tqdm(total=total_windows)
        pbar.set_description(f"Processing SDD dataset {os.path.basename(annotations_file)}")

        for start_idx in range(0, len(all_frames) - self.seq_len + 1, self.skip):
            window_frames = all_frames[start_idx : start_idx + self.seq_len]
            frame_to_idx = {f: i for i, f in enumerate(window_frames)}

            valid_track_ids: List[int] = []
            for tid, traj in trajectories.items():
                traj_frames = {f for f, _, _ in traj}
                if all(f in traj_frames for f in window_frames):
                    valid_track_ids.append(tid)

            if len(valid_track_ids) < 2:
                pbar.update(1)
                continue

            num_peds = len(valid_track_ids)
            positions = np.zeros((num_peds, 2, self.seq_len), dtype=np.float32)
            rel_positions = np.zeros((num_peds, 2, self.seq_len), dtype=np.float32)

            for i, tid in enumerate(valid_track_ids):
                traj = trajectories[tid]
                frame_to_pos = {f: (x, y) for f, x, y in traj}
                coords = np.zeros((2, self.seq_len), dtype=np.float32)
                for f in window_frames:
                    x, y = frame_to_pos[f]
                    coords[0, frame_to_idx[f]] = x
                    coords[1, frame_to_idx[f]] = y

                positions[i] = coords
                rel = np.zeros_like(coords)
                rel[:, 1:] = coords[:, 1:] - coords[:, :-1]
                rel_positions[i] = rel

            obs_abs = torch.from_numpy(positions[:, :, : self.obs_len])
            pred_abs = torch.from_numpy(positions[:, :, self.obs_len :])
            obs_rel = torch.from_numpy(rel_positions[:, :, : self.obs_len])
            pred_rel = torch.from_numpy(rel_positions[:, :, self.obs_len :])

            non_linear_ped = torch.zeros(num_peds, dtype=torch.float32)
            loss_mask = torch.ones((num_peds, self.seq_len), dtype=torch.float32)

            V_obs, A_obs = seq_to_graph(obs_abs, obs_rel)
            V_pred, A_pred = seq_to_graph(pred_abs, pred_rel)

            self.obs_traj_list.append(obs_abs)
            self.pred_traj_list.append(pred_abs)
            self.obs_traj_rel_list.append(obs_rel)
            self.pred_traj_rel_list.append(pred_rel)
            self.non_linear_ped_list.append(non_linear_ped)
            self.loss_mask_list.append(loss_mask)
            self.V_obs_list.append(V_obs)
            self.A_obs_list.append(A_obs)
            self.V_pred_list.append(V_pred)
            self.A_pred_list.append(A_pred)

            pbar.update(1)

        pbar.close()


