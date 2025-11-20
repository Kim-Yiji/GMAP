import os
import math
import shlex
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .affordance import compute_threat_scores
from .labels import load_track_labels, load_class_sizes


SDD_PEDESTRIANS = {"Pedestrian"}
SDD_OBJECTS = {"Skater", "Biker", "Car", "Bus", "Cart"}


def anorm(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def read_file_ext(_path, delim=None):
    """Read SDD annotation file.
    
    Supports two formats:
    1. SDD original format (space-delimited): frame_id track_id xmin ymin xmax ymax frame? lost occluded generated "label"
    2. ETH/UCY format (tab-delimited): frame_id track_id x y [type] [bbox_w] [bbox_h]
    """
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try to detect format: SDD format has quotes around label, ETH/UCY uses tabs
            try:
                if delim is None:
                    # Auto-detect: if has quotes, use shlex to properly parse; otherwise try tab
                    if '"' in line or "'" in line:
                        # Use shlex to properly handle quoted strings
                        parts = shlex.split(line)
                    else:
                        parts = line.split('\t') if '\t' in line else line.split()
                else:
                    if delim == '\t':
                        parts = line.split('\t')
                    else:
                        # For space delimiter with quotes, use shlex
                        parts = shlex.split(line) if ('"' in line or "'" in line) else line.split()
            except ValueError:
                # If shlex fails, fall back to simple split
                parts = line.split('\t') if '\t' in line else line.split()
            
            try:
                # Try SDD format first (frame_id track_id xmin ymin xmax ymax frame? lost occluded generated "label")
                # SDD format has 10 fields (0-9 indices) with label at the end
                if len(parts) >= 10:
                    # SDD format: extract bbox and convert to center point
                    frame = float(parts[0])
                    tid = float(parts[1])
                    xmin = float(parts[2])
                    ymin = float(parts[3])
                    xmax = float(parts[4])
                    ymax = float(parts[5])
                    # Convert bbox to center point
                    x = (xmin + xmax) / 2.0
                    y = (ymin + ymax) / 2.0
                    # Extract label (last part, already unquoted by shlex)
                    label = parts[-1] if len(parts) > 0 else 'Pedestrian'
                    # Calculate bbox area
                    area = (xmax - xmin) * (ymax - ymin)
                    data.append([frame, tid, x, y, label, area])
                elif len(parts) >= 4:
                    # ETH/UCY format: <frame_id> <track_id> <x> <y> [type] [bbox_w] [bbox_h]
                    frame = float(parts[0])
                    tid = float(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    typ = parts[4] if len(parts) > 4 else 'Pedestrian'
                    # Remove quotes if present
                    typ = typ.strip('"').strip("'")
                    bw = float(parts[5]) if len(parts) > 5 else 0.0
                    bh = float(parts[6]) if len(parts) > 6 else 0.0
                    area = bw * bh if (bw > 0 and bh > 0) else 0.0
                    data.append([frame, tid, x, y, typ, area])
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
    
    return np.asarray(data, dtype=object)


class SDDTrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, delim='\t', labels_dir=None, class_size_csv=None):
        super().__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.skip = skip
        self.delim = delim
        self.labels_dir = labels_dir
        self.class_sizes = load_class_sizes(class_size_csv)

        all_files = sorted(os.listdir(self.data_dir))
        # Filter to only text files (.txt, .csv) or annotations.txt files
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files 
                     if os.path.isfile(os.path.join(self.data_dir, _path)) and 
                     (_path.endswith('.txt') or _path.endswith('.csv') or _path == 'annotations.txt')]

        self.V_obs, self.A_obs, self.V_pred, self.A_pred = [], [], [], []
        self.obs_traj, self.pred_traj, self.obs_traj_rel, self.pred_traj_rel = [], [], [], []
        self.track_ids_list = []  # Store track IDs for each sequence

        pbar = tqdm(total=len(all_files))
        pbar.set_description('Processing SDD ' + self.data_dir.split('/')[-2])

        for path in all_files:
            base = os.path.splitext(os.path.basename(path))[0]
            # Prefer ETH/UCY style (<frame> <id> <x> <y>) if present; fallback to extended
            try:
                data = read_file_ext(path, self.delim)
            except Exception:
                data = None

            # Load label mapping per video if labels_dir is provided
            label_map = {}
            if self.labels_dir:
                label_map = load_track_labels(self.labels_dir, base)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                tids = np.unique(curr_seq_data[:, 1])

                # Build per-track arrays
                positions = np.zeros((len(tids), 2, self.seq_len))
                rel_positions = np.zeros((len(tids), 2, self.seq_len))
                areas = np.zeros((len(tids),), dtype=float)
                is_ped = np.zeros((len(tids),), dtype=bool)
                is_obj = np.zeros((len(tids),), dtype=bool)

                num_tracks = 0
                for _, tid in enumerate(tids):
                    seg = curr_seq_data[curr_seq_data[:, 1] == tid]
                    # ensure full window
                    pad_front = frames.index(seg[0, 0]) - idx
                    pad_end = frames.index(seg[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    coords = np.transpose(np.array(seg[:, 2:4], dtype=float))
                    rel = np.zeros_like(coords)
                    rel[:, 1:] = coords[:, 1:] - coords[:, :-1]
                    positions[num_tracks, :, pad_front:pad_end] = coords
                    rel_positions[num_tracks, :, pad_front:pad_end] = rel
                    # Size/type resolution:
                    cls_name = None
                    if seg.shape[1] > 5:  # extended format with type/boxes
                        areas[num_tracks] = float(seg[-1, 5]) if seg.shape[1] > 5 else 0.0
                        t = str(seg[-1, 4]) if seg.shape[1] > 4 else 'Pedestrian'
                        cls_name = t
                    else:
                        # ETH/UCY format; use labels_dir + class_size_csv
                        tid_int = int(seg[-1, 1])
                        cls_name = label_map.get(tid_int, 'Pedestrian')
                        # derive size from class mapping; default 0 if unknown
                        areas[num_tracks] = float(self.class_sizes.get(cls_name, 0.0))

                    is_ped[num_tracks] = (cls_name in SDD_PEDESTRIANS)
                    is_obj[num_tracks] = (cls_name in SDD_OBJECTS)
                    num_tracks += 1

                if num_tracks < 2:
                    continue

                positions = positions[:num_tracks]
                rel_positions = rel_positions[:num_tracks]
                areas = areas[:num_tracks]
                is_ped = is_ped[:num_tracks]
                is_obj = is_obj[:num_tracks]
                
                # Store track IDs for this sequence
                sequence_track_ids = [int(tid) for tid in tids[:num_tracks]]

                # Convert to torch
                obs_abs = torch.from_numpy(positions[:, :, :self.obs_len]).float()
                pred_abs = torch.from_numpy(positions[:, :, self.obs_len:]).float()
                obs_rel = torch.from_numpy(rel_positions[:, :, :self.obs_len]).float()
                pred_rel = torch.from_numpy(rel_positions[:, :, self.obs_len:]).float()

                V_obs = torch.zeros((self.obs_len, num_tracks, 2), dtype=torch.float)
                V_pred = torch.zeros((self.pred_len, num_tracks, 2), dtype=torch.float)
                A_disp = torch.zeros((self.obs_len, num_tracks, num_tracks), dtype=torch.float)
                A_dist = torch.zeros((self.obs_len, num_tracks, num_tracks), dtype=torch.float)
                A_pp_threat = torch.zeros_like(A_dist)
                A_po_threat = torch.zeros_like(A_dist)

                ped_mask = torch.from_numpy(is_ped)
                obj_mask = torch.from_numpy(is_obj)
                sizes = torch.from_numpy(areas).float()

                for t in range(self.obs_len):
                    V_obs[t] = obs_rel[:, :, t]
                    for i in range(num_tracks):
                        for j in range(i + 1, num_tracks):
                            A_dist[t, i, j] = A_dist[t, j, i] = torch.dist(obs_abs[i, :, t], obs_abs[j, :, t])
                            A_disp[t, i, j] = A_disp[t, j, i] = torch.dist(obs_rel[i, :, t], obs_rel[j, :, t])

                    pos_t = obs_abs[:, :, t]
                    vel_t = obs_rel[:, :, t]
                    T_pp, T_po = compute_threat_scores(pos_t, vel_t, sizes, ped_mask, obj_mask)
                    A_pp_threat[t] = T_pp
                    A_po_threat[t] = T_po

                for t in range(self.pred_len):
                    V_pred[t] = pred_rel[:, :, t]

                A_obs = torch.stack([A_disp, A_dist, A_pp_threat, A_po_threat], dim=0)

                self.V_obs.append(V_obs.clone())
                self.A_obs.append(A_obs.clone())
                self.V_pred.append(V_pred.clone())
                # For completeness in evaluation; not used by training loss here
                self.A_pred.append(A_obs.clone())
                self.obs_traj.append(obs_abs)
                self.pred_traj.append(pred_abs)
                self.obs_traj_rel.append(obs_rel)
                self.pred_traj_rel.append(pred_rel)
                self.track_ids_list.append(sequence_track_ids)

            pbar.update(1)

        pbar.close()

    def __len__(self):
        return len(self.V_obs)

    def __getitem__(self, index):
        return [
            self.obs_traj[index], self.pred_traj[index],
            self.obs_traj_rel[index], self.pred_traj_rel[index],
            torch.tensor(0.0), torch.ones((self.obs_traj[index].size(0), self.obs_traj[index].size(2) + self.pred_traj[index].size(2))),
            self.V_obs[index], self.A_obs[index],
            self.V_pred[index], self.A_pred[index],
            self.track_ids_list[index]  # Add track IDs
        ]
    
    def get_track_ids(self, index):
        """Get track IDs for a specific sequence."""
        return self.track_ids_list[index]


