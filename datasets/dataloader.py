# Enhanced Dataloader for DMRGCN + GP-Graph integration
# Based on Social-GAN dataloader with advanced caching system

import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from .cache_manager import CacheManager


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


def poly_fit(traj, traj_len, threshold):
    """Check if trajectory is linear using polynomial fitting"""
    if len(traj) < 2:
        return 0
    
    # Fit polynomial of degree 1 (linear)
    x = np.arange(len(traj))
    y = traj
    
    # Calculate linear fit
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    
    # Calculate error
    error = np.sum((y - poly(x)) ** 2)
    
    return 1 if error > threshold else 0


def read_file(_path, delim='\t'):
    """Read trajectory data from file"""
    data = []
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def seq_to_graph(seq, seq_rel, agent_ids):
    """Convert sequence to graph representation"""
    # Build distance and displacement matrices
    dist_matrix = build_pairwise_distance_matrix(seq)
    disp_matrix = build_pairwise_displacement_matrix(seq_rel)
    
    # Create adjacency matrix based on distance threshold
    threshold = 2.0  # meters
    adj_matrix = (dist_matrix < threshold).float()
    
    # Remove self-loops
    for t in range(adj_matrix.shape[0]):
        adj_matrix[t].fill_diagonal_(0)
    
    # Stack matrices: [distance, displacement, adjacency]
    V = torch.stack([seq, seq_rel], dim=0)  # (2, T, N, 2)
    A = torch.stack([dist_matrix, disp_matrix, adj_matrix], dim=0)  # (3, T, N, N)
    
    return V, A, agent_ids


class TrajectoryDataset(Dataset):
    """Enhanced Trajectory Dataset with advanced caching system"""

    def __init__(self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1, delim='\t', 
                 use_cache=True, cache_dir='./data_cache', force_rebuild=False):
        """
        Args:
        - data_dir: Directory containing dataset files
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        - use_cache: Whether to use cached preprocessed data
        - cache_dir: Directory to store cached data
        - force_rebuild: Force rebuild cache even if valid cache exists
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.use_cache = use_cache
        self.force_rebuild = force_rebuild
        
        # Initialize cache manager
        self.cache_manager = CacheManager(cache_dir)
        
        # Data parameters for cache key generation
        self.data_params = {
            'obs_len': obs_len,
            'pred_len': pred_len,
            'skip': skip,
            'delim': delim,
            'threshold': threshold,
            'min_ped': min_ped
        }
        
        # Generate cache key and path
        cache_key = self.cache_manager.generate_cache_key(data_dir, self.data_params)
        cache_path = self.cache_manager.get_cache_path(cache_key)
        
        # Try to load from cache or process from scratch
        if self.use_cache and not force_rebuild:
            is_valid, reason = self.cache_manager.is_cache_valid(cache_path, data_dir, self.data_params)
            
            if is_valid:
                print(f"🚀 Loading preprocessed data from cache: {cache_path}")
                cached_data = self.cache_manager.load_cache(cache_path)
                
                if cached_data is not None:
                    # Normalize data types and convert to tensors
                    normalized_data = self.cache_manager.normalize_data_types(cached_data)
                    processed_data = self.cache_manager.convert_to_tensors(normalized_data, obs_len, pred_len)
                    
                    # Set attributes
                    self._set_attributes_from_data(processed_data)
                    return
                else:
                    print("⚠️  Cache loading failed, processing from scratch...")
            else:
                print(f"⚠️  Cache invalid ({reason}), processing from scratch...")
        
        # Process data from scratch
        print(f"🔄 Processing data from scratch...")
        self._process_data_from_scratch(threshold, min_ped)
        
        # Save to cache
        if self.use_cache:
            print(f"💾 Saving preprocessed data to cache...")
            self._save_to_cache(cache_path)
    
    def _set_attributes_from_data(self, data):
        """Set dataset attributes from processed data"""
        self.num_seq = data['num_seq']
        self.max_peds_in_frame = data['max_peds_in_frame']
        
        # Trajectory data
        self.obs_traj = data['obs_traj']
        self.pred_traj = data['pred_traj']
        self.obs_traj_rel = data['obs_traj_rel']
        self.pred_traj_rel = data['pred_traj_rel']
        self.loss_mask = data['loss_mask']
        self.non_linear_ped = data['non_linear_ped']
        self.agent_ids = data['agent_ids']
        self.seq_start_end = data['seq_start_end']
        
        # Graph data
        self.V_obs = data['V_obs']
        self.A_obs = data['A_obs']
        self.V_pred = data['V_pred']
        self.A_pred = data['A_pred']
        self.agent_ids_per_seq = data['agent_ids_per_seq']
    
    def _process_data_from_scratch(self, threshold=0.002, min_ped=1):
        """Process data from scratch"""
        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        agent_ids_list = []
        loss_mask_list = []
        non_linear_ped = []
        
        self.max_peds_in_frame = 0
        
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
                    curr_agent_ids[_idx] = ped_id

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

        # Convert to tensors
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.agent_ids = torch.from_numpy(agent_ids_list).type(torch.long)
        
        # Create sequence start/end indices
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Convert Trajectories to Enhanced Graphs
        self.V_obs = []
        self.A_obs = []
        self.V_pred = []
        self.A_pred = []
        self.agent_ids_per_seq = []

        pbar = tqdm(total=len(self.seq_start_end))
        pbar.set_description(f'Processing {self.data_dir.split("/")[-3]} dataset {self.data_dir.split("/")[-2]}')

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
    
    def _save_to_cache(self, cache_path):
        """Save processed data to cache"""
        # Prepare data for caching (convert tensors to numpy for storage)
        cache_data = {
            'num_seq': self.num_seq,
            'max_peds_in_frame': self.max_peds_in_frame,
            'seq_list': self.obs_traj.numpy() if hasattr(self.obs_traj, 'numpy') else self.obs_traj,
            'seq_list_rel': self.obs_traj_rel.numpy() if hasattr(self.obs_traj_rel, 'numpy') else self.obs_traj_rel,
            'agent_ids_list': self.agent_ids.numpy() if hasattr(self.agent_ids, 'numpy') else self.agent_ids,
            'loss_mask_list': self.loss_mask.numpy() if hasattr(self.loss_mask, 'numpy') else self.loss_mask,
            'non_linear_ped': self.non_linear_ped.numpy() if hasattr(self.non_linear_ped, 'numpy') else self.non_linear_ped,
            'num_peds_in_seq': [end - start for start, end in self.seq_start_end],
            'V_obs': self.V_obs,
            'A_obs': self.A_obs,
            'V_pred': self.V_pred,
            'A_pred': self.A_pred,
            'agent_ids_per_seq': self.agent_ids_per_seq
        }
        
        self.cache_manager.save_cache(cache_path, cache_data, self.data_params)

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
            self.agent_ids_per_seq[index]
        ]
        return out


def collate_fn(batch):
    """Custom collate function to handle variable-sized sequences"""
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