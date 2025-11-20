# Dataloader code based on Social-GAN
# https://github.com/agrimgupta92/sgan/blob/master/sgan/data/trajectories.py

import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .threat_score import compute_threat_score_batch


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return NORM


def seq_to_graph(seq, seq_rel):
    assert seq.shape == seq_rel.shape

    num_nodes = seq.shape[0]
    seq_len = seq.shape[2]

    V = torch.zeros((seq_len, num_nodes, 2), dtype=torch.float)
    A_dist = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    A_disp = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)

    # 기본 distance / relative displacement relation 그래프 생성
    for t in range(seq_len):
        for n in range(num_nodes):
            V[t, n, :] = seq_rel[n, :, t]
            for l in range(n + 1, num_nodes):
                # distance relation (A_dist)
                A_dist[t, n, l] = A_dist[t, l, n] = anorm(seq[n, :, t], seq[l, :, t])
                # relative displacement relation (A_disp)
                A_disp[t, n, l] = A_disp[t, l, n] = anorm(seq_rel[n, :, t], seq_rel[l, :, t])

    # ------------------------------------------------------------
    # Threat relation 그래프 생성
    # ------------------------------------------------------------
    # seq: (num_nodes, 2, seq_len), seq_rel: (num_nodes, 2, seq_len)
    # compute_threat_score_batch는 T_ij (pp/po 모두)를 [0, 1] 범위로 반환
    # 여기서 threat score를 adjacency weight로 직접 사용한다.
    threat_score, _ = compute_threat_score_batch(
        obs_traj=seq,
        obs_traj_rel=seq_rel,
        obstacle_sizes=None,
        weights=None,
        tau=0.15,
        beta=0.5,
        pedestrian_mask=None,
        object_labels=None,
    )
    # threat_score: (num_nodes, num_nodes, seq_len) → (seq_len, num_nodes, num_nodes)
    A_threat = threat_score.permute(2, 0, 1).contiguous()

    # relation 차원: [A_disp, A_dist, A_threat]
    return V, torch.stack([A_disp, A_dist, A_threat], dim=0)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    # 궤적이 너무 짧으면 선형으로 간주
    if traj_len < 3:
        return 0.0
        
    t = np.linspace(0, traj_len - 1, traj_len)
    try:
        res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
        res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
        
        # res가 빈 배열인 경우 처리
        if len(res_x) == 0 or len(res_y) == 0:
            return 0.0
            
        if res_x[0] + res_y[0] >= threshold:
            return 1.0
        else:
            return 0.0
    except:
        # 오류 발생 시 선형으로 간주
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=0, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = sorted(os.listdir(self.data_dir))
        # Only process plain text trajectory files
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files if _path.endswith('.txt')]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))

                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))

                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    
                    ## 스탠포드 데이터로 추가된 부분
                    # Limit sequence length to prevent memory issues
                    max_seq_len = 100  # Maximum sequence length
                    if len(curr_ped_seq) > max_seq_len:
                        curr_ped_seq = curr_ped_seq[:max_seq_len]
                    
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    
                    ## 스탠포드 데이터로 추가된 부분
                    # 20프레임 이상이면 20프레임만 사용
                    if curr_ped_seq.shape[1] >= self.seq_len:
                        curr_ped_seq = curr_ped_seq[:, :self.seq_len]
                    else:
                        continue
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered

                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy matrix to torch tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Convert Trajectories to Graphs
        self.V_obs = []
        self.A_obs = []
        self.V_pred = []
        self.A_pred = []

        pbar = tqdm(total=len(self.seq_start_end))
        pbar.set_description(
            'Processing {0} dataset {1}'.format(self.data_dir.split('/')[-3], self.data_dir.split('/')[-2]))

        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :])
            self.V_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :])
            self.V_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
            pbar.update(1)
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.V_obs[index], self.A_obs[index],
            self.V_pred[index], self.A_pred[index]
        ]
        return out


class CachedTrajectoryDataset(Dataset):
    """Dataloader for preprocessed .pt cache files"""

    def __init__(self, cache_file_path):
        """
        Args:
        - cache_file_path: Path to the preprocessed .pt cache file
        """
        super(CachedTrajectoryDataset, self).__init__()
        
        print(f"Loading cached data from {cache_file_path}...")
        cached_data = torch.load(cache_file_path, map_location='cpu')
        
        # Load all required tensors and lists
        self.obs_traj = cached_data['obs_traj']
        self.pred_traj = cached_data['pred_traj']
        self.obs_traj_rel = cached_data['obs_traj_rel']
        self.pred_traj_rel = cached_data['pred_traj_rel']
        self.loss_mask = cached_data['loss_mask']
        self.non_linear_ped = cached_data['non_linear_ped']
        self.seq_start_end = cached_data['seq_start_end']
        self.V_obs = cached_data['V_obs']
        self.A_obs = cached_data['A_obs']
        self.V_pred = cached_data['V_pred']
        self.A_pred = cached_data['A_pred']
        
        self.num_seq = len(self.seq_start_end)
        
        print(f"Loaded {self.num_seq} sequences with {len(self.obs_traj)} total pedestrian trajectories")

        # ------------------------------------------------------------
        # Threat relation을 포함하는 최신 그래프 구조로 재계산
        # (기존 캐시의 A_obs/A_pred는 distance/disp 2개 relation만 포함할 수 있으므로)
        # ------------------------------------------------------------
        self.V_obs = []
        self.A_obs = []
        self.V_pred = []
        self.A_pred = []

        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            # 관찰 구간 그래프 (obs_len 프레임만 사용)
            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :])
            self.V_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            # 예측 구간 그래프 (pred_len 프레임만 사용)
            v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :])
            self.V_pred.append(v_.clone())
            self.A_pred.append(a_.clone())

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.V_obs[index], self.A_obs[index],
            self.V_pred[index], self.A_pred[index]
        ]
        return out
