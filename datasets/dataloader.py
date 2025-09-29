# Enhanced Dataloader for DMRGCN + GP-Graph integration
# Based on Social-GAN dataloader with additional features for group-aware modeling

import os
import math
import pickle
import hashlib
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


def anorm(p1, p2):
    """Calculate euclidean distance between two points"""
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return NORM


def build_pairwise_distance_matrix(seq):
    """Build pairwise distance matrix for all time steps
    
    Args:
        seq: (N, 2, T) - absolute positions
    
    Returns:
        dist_matrix: (T, N, N) - pairwise distance matrix
    """
    num_nodes = seq.shape[0]
    seq_len = seq.shape[2]
    
    dist_matrix = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    
    for t in range(seq_len):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist_matrix[t, i, j] = anorm(seq[i, :, t], seq[j, :, t])
    
    return dist_matrix


def build_pairwise_displacement_matrix(seq_rel):
    """Build pairwise relative displacement matrix for all time steps
    
    Args:
        seq_rel: (N, 2, T) - relative displacements
    
    Returns:
        disp_matrix: (T, N, N) - pairwise displacement matrix
    """
    num_nodes = seq_rel.shape[0]
    seq_len = seq_rel.shape[2]
    
    disp_matrix = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    
    for t in range(seq_len):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    disp_matrix[t, i, j] = anorm(seq_rel[i, :, t], seq_rel[j, :, t])
    
    return disp_matrix


def seq_to_graph(seq, seq_rel, agent_ids=None):
    """Convert trajectory sequences to graph representation
    
    Args:
        seq: (N, 2, T) - absolute positions  
        seq_rel: (N, 2, T) - relative displacements
        agent_ids: (N,) - agent IDs for tracking
    
    Returns:
        V: (T, N, 2) - node features (relative displacements)
        A: (2, T, N, N) - adjacency matrices [displacement, distance]
        agent_ids: (N,) - preserved agent IDs
    """
    assert seq.shape == seq_rel.shape
    
    num_nodes = seq.shape[0]
    seq_len = seq.shape[2]
    
    # Node features: relative displacements
    V = torch.zeros((seq_len, num_nodes, 2), dtype=torch.float)
    for t in range(seq_len):
        for n in range(num_nodes):
            V[t, n, :] = seq_rel[n, :, t]
    
    # Adjacency matrices
    A_dist = build_pairwise_distance_matrix(seq)
    A_disp = build_pairwise_displacement_matrix(seq_rel)
    
    # Stack: [displacement, distance]
    A = torch.stack([A_disp, A_dist], dim=0)
    
    # Preserve agent IDs if provided
    if agent_ids is None:
        agent_ids = torch.arange(num_nodes, dtype=torch.long)
    
    return V, A, agent_ids


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
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
    """Enhanced Trajectory Dataset for DMRGCN + GP-Graph with caching support"""

    def __init__(self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1, delim='\t', 
                 use_cache=True, cache_dir='./data_cache'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        - use_cache: Whether to use cached preprocessed data
        - cache_dir: Directory to store cached data
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate unique cache key based on data parameters
        cache_key = self._generate_cache_key()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Try to load from cache first
        if self.use_cache and os.path.exists(cache_path):
            print(f"🚀 Loading preprocessed data from cache: {cache_path}")
            self._load_from_cache(cache_path)
        else:
            print(f"🔄 Processing data from scratch...")
            self._process_data_from_scratch()
            
            # Save to cache
            if self.use_cache:
                print(f"💾 Saving preprocessed data to cache: {cache_path}")
                self._save_to_cache(cache_path)

    def _generate_cache_key(self):
        """Generate unique cache key based on data directory and parameters"""
        # Get data directory name and modification times of all files
        data_dir_name = os.path.basename(os.path.normpath(self.data_dir))
        
        # Include all files and their modification times
        all_files = sorted(os.listdir(self.data_dir))
        file_info = []
        for fname in all_files:
            fpath = os.path.join(self.data_dir, fname)
            if os.path.isfile(fpath):
                mtime = os.path.getmtime(fpath)
                file_info.append((fname, mtime))
        
        # Create hash from parameters and file info
        params_str = f"{data_dir_name}_{self.obs_len}_{self.pred_len}_{self.skip}_{self.delim}"
        files_str = str(file_info)
        combined_str = params_str + files_str
        
        return hashlib.md5(combined_str.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_path):
        """Load preprocessed data from cache"""
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        self.num_seq = cached_data['num_seq']
        self.seq_list = cached_data['seq_list']
        self.seq_list_rel = cached_data['seq_list_rel']
        self.agent_ids_list = cached_data['agent_ids_list']
        self.loss_mask_list = cached_data['loss_mask_list']
        self.non_linear_ped = cached_data['non_linear_ped']
        self.max_peds_in_frame = cached_data['max_peds_in_frame']
        
        # Handle backward compatibility for num_peds_in_seq
        if 'num_peds_in_seq' in cached_data:
            self.num_peds_in_seq = cached_data['num_peds_in_seq']
        else:
            # Reconstruct num_peds_in_seq from existing data
            print("⚠️  Reconstructing num_peds_in_seq from cached data...")
            self.num_peds_in_seq = []
            current_idx = 0
            for i in range(self.num_seq):
                # Find the number of pedestrians in this sequence
                seq_end = current_idx
                while seq_end < len(self.agent_ids_list) and self.agent_ids_list[seq_end] != 0:
                    seq_end += 1
                num_peds = seq_end - current_idx
                self.num_peds_in_seq.append(num_peds)
                current_idx = seq_end
        
        # Precomputed graph data
        self.V_obs = cached_data['V_obs']
        self.A_obs = cached_data['A_obs']
        self.V_pred = cached_data['V_pred']
        self.A_pred = cached_data['A_pred']
        
        # Convert numpy matrix to torch tensor (same as in _process_data_from_scratch)
        self.obs_traj = torch.from_numpy(self.seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(self.seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(self.loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(self.non_linear_ped).type(torch.float)
        self.agent_ids = torch.from_numpy(self.agent_ids_list).type(torch.long)
        
        # Reconstruct seq_start_end from cached data
        cum_start_idx = [0] + np.cumsum(self.num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        
        print(f"✅ Loaded {self.num_seq} sequences from cache")
    
    def _save_to_cache(self, cache_path):
        """Save preprocessed data to cache"""
        cached_data = {
            'num_seq': self.num_seq,
            'seq_list': self.seq_list,
            'seq_list_rel': self.seq_list_rel,
            'agent_ids_list': self.agent_ids_list,
            'loss_mask_list': self.loss_mask_list,
            'non_linear_ped': self.non_linear_ped,
            'max_peds_in_frame': self.max_peds_in_frame,
            'num_peds_in_seq': self.num_peds_in_seq,
            'V_obs': self.V_obs,
            'A_obs': self.A_obs,
            'V_pred': self.V_pred,
            'A_pred': self.A_pred
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        
        print(f"💾 Cached {self.num_seq} sequences to {cache_path}")
    
    def _process_data_from_scratch(self):
        """Process data from scratch (original processing logic)"""
        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        agent_ids_list = []  # Track agent IDs
        loss_mask_list = []
        non_linear_ped = []
        
        # Default threshold and min_ped for processing
        threshold = 0.002
        min_ped = 1
        
        for path in all_files:
            data = read_file(path, self.delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))

                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                curr_agent_ids = np.zeros(len(peds_in_curr_seq), dtype=int)

                num_peds_considered = 0
                _non_linear_ped = []
                for ped_idx, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])

                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered

                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_agent_ids[_idx] = ped_id  # Store agent ID

                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, self.pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    agent_ids_list.append(curr_agent_ids[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        agent_ids_list = np.concatenate(agent_ids_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy matrix to torch tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.agent_ids = torch.from_numpy(agent_ids_list).type(torch.long)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Store data for caching  
        self.seq_list = seq_list
        self.seq_list_rel = seq_list_rel
        self.agent_ids_list = agent_ids_list
        self.loss_mask_list = loss_mask_list
        self.num_peds_in_seq = num_peds_in_seq

        # Convert Trajectories to Enhanced Graphs
        self.V_obs = []
        self.A_obs = []
        self.V_pred = []
        self.A_pred = []
        self.agent_ids_per_seq = []

        pbar = tqdm(total=len(self.seq_start_end))
        pbar.set_description(
            'Processing {0} dataset {1}'.format(self.data_dir.split('/')[-3], self.data_dir.split('/')[-2]))

        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            seq_agent_ids = self.agent_ids[start:end]
            
            # Observation phase
            v_, a_, agent_ids_ = seq_to_graph(
                self.obs_traj[start:end, :], 
                self.obs_traj_rel[start:end, :],
                seq_agent_ids
            )
            self.V_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            
            # Prediction phase
            v_, a_, _ = seq_to_graph(
                self.pred_traj[start:end, :], 
                self.pred_traj_rel[start:end, :],
                seq_agent_ids
            )
            self.V_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
            self.agent_ids_per_seq.append(agent_ids_.clone())
            
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
            self.V_pred[index], self.A_pred[index],
            self.agent_ids_per_seq[index]  # Add agent IDs
        ]
        return out


def collate_fn(batch):
    """Custom collate function to handle variable-sized sequences
    while preserving agent IDs and enhanced graph information
    """
    # Unpack batch
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, 
     loss_mask, V_obs, A_obs, V_pred, A_pred, agent_ids) = zip(*batch)
    
    # Standard collation
    _len = [len(seq) for seq in obs_traj]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = torch.LongTensor([[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])])
    
    # Concatenate trajectory data
    obs_traj = torch.cat(obs_traj, dim=0).permute(2, 0, 1)  # (T, N, 2)
    pred_traj = torch.cat(pred_traj, dim=0).permute(2, 0, 1)  # (T, N, 2)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=0).permute(2, 0, 1)  # (T, N, 2)
    pred_traj_rel = torch.cat(pred_traj_rel, dim=0).permute(2, 0, 1)  # (T, N, 2)
    non_linear_ped = torch.cat(non_linear_ped, dim=0)
    loss_mask = torch.cat(loss_mask, dim=0).permute(1, 0)  # (T, N)
    agent_ids = torch.cat(agent_ids, dim=0)  # (N,)
    
    # Handle graph data - pad to maximum size in batch
    max_nodes = max([V.shape[1] for V in V_obs])
    
    V_obs_batch = []
    A_obs_batch = []
    V_pred_batch = []
    A_pred_batch = []
    
    for i, (v_obs, a_obs, v_pred, a_pred) in enumerate(zip(V_obs, A_obs, V_pred, A_pred)):
        n_nodes = v_obs.shape[1]
        
        # Pad V (T, N, 2) -> (T, max_N, 2)
        v_obs_pad = torch.zeros(v_obs.shape[0], max_nodes, v_obs.shape[2])
        v_obs_pad[:, :n_nodes, :] = v_obs
        V_obs_batch.append(v_obs_pad)
        
        v_pred_pad = torch.zeros(v_pred.shape[0], max_nodes, v_pred.shape[2])
        v_pred_pad[:, :n_nodes, :] = v_pred
        V_pred_batch.append(v_pred_pad)
        
        # Pad A (R, T, N, N) -> (R, T, max_N, max_N)
        a_obs_pad = torch.zeros(a_obs.shape[0], a_obs.shape[1], max_nodes, max_nodes)
        a_obs_pad[:, :, :n_nodes, :n_nodes] = a_obs
        A_obs_batch.append(a_obs_pad)
        
        a_pred_pad = torch.zeros(a_pred.shape[0], a_pred.shape[1], max_nodes, max_nodes)
        a_pred_pad[:, :, :n_nodes, :n_nodes] = a_pred
        A_pred_batch.append(a_pred_pad)
    
    V_obs_batch = torch.stack(V_obs_batch, dim=0)  # (B, T, max_N, 2)
    A_obs_batch = torch.stack(A_obs_batch, dim=0)  # (B, R, T, max_N, max_N)
    V_pred_batch = torch.stack(V_pred_batch, dim=0)  # (B, T, max_N, 2)
    A_pred_batch = torch.stack(A_pred_batch, dim=0)  # (B, R, T, max_N, max_N)
    
    return [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask,
            V_obs_batch, A_obs_batch, V_pred_batch, A_pred_batch, seq_start_end, agent_ids]
