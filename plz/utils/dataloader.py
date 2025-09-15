import os
import math
import torch
try:
    import numpy as np
except ImportError:
    print("Warning: NumPy not available. Some functionality may be limited.")
    np = None
from tqdm import tqdm
from torch.utils.data import Dataset


def anorm(p1, p2):
    """Calculate Euclidean distance between two points"""
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return NORM


def calculate_velocity(seq):
    """Calculate velocity from position sequence"""
    if seq.shape[2] <= 1:
        return np.zeros_like(seq)
    
    velocity = np.zeros_like(seq)
    velocity[:, :, 1:] = seq[:, :, 1:] - seq[:, :, :-1]
    return velocity


def calculate_acceleration(velocity):
    """Calculate acceleration from velocity sequence"""
    if velocity.shape[2] <= 1:
        return np.zeros_like(velocity)
    
    acceleration = np.zeros_like(velocity)
    acceleration[:, :, 1:] = velocity[:, :, 1:] - velocity[:, :, :-1]
    return acceleration


def seq_to_graph(seq, seq_rel, include_velocity=True, include_acceleration=True):
    """
    Convert sequence data to graph format with enhanced features
    
    Args:
        seq: Absolute trajectory sequence [num_ped, 2, seq_len]
        seq_rel: Relative trajectory sequence [num_ped, 2, seq_len]
        include_velocity: Whether to include velocity in graph
        include_acceleration: Whether to include acceleration in graph
    """
    assert seq.shape == seq_rel.shape

    num_nodes = seq.shape[0]
    seq_len = seq.shape[2]

    # Basic graph construction
    V = torch.zeros((seq_len, num_nodes, 2), dtype=torch.float)
    A_dist = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    A_disp = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    
    # Enhanced adjacency matrices
    A_vel = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    A_acc = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)

    # Calculate velocity and acceleration
    velocity = calculate_velocity(seq_rel)
    acceleration = calculate_acceleration(velocity)

    for t in range(seq_len):
        for n in range(num_nodes):
            V[t, n, :] = seq_rel[n, :, t]
            
            for l in range(n + 1, num_nodes):
                # Distance-based adjacency
                A_dist[t, n, l] = A_dist[t, l, n] = anorm(seq[n, :, t], seq[l, :, t])
                
                # Displacement-based adjacency
                A_disp[t, n, l] = A_disp[t, l, n] = anorm(seq_rel[n, :, t], seq_rel[l, :, t])
                
                if include_velocity:
                    # Velocity-based adjacency
                    A_vel[t, n, l] = A_vel[t, l, n] = anorm(velocity[n, :, t], velocity[l, :, t])
                
                if include_acceleration:
                    # Acceleration-based adjacency
                    A_acc[t, n, l] = A_acc[t, l, n] = anorm(acceleration[n, :, t], acceleration[l, :, t])

    # Stack adjacency matrices
    adjacency_matrices = [A_disp, A_dist]
    if include_velocity:
        adjacency_matrices.append(A_vel)
    if include_acceleration:
        adjacency_matrices.append(A_acc)

    return V, torch.stack(adjacency_matrices, dim=0)


def poly_fit(traj, traj_len, threshold):
    """
    Determine if trajectory is linear or non-linear
    
    Args:
        traj: Trajectory array of shape (2, traj_len)
        traj_len: Length of trajectory
        threshold: Threshold for non-linearity
    
    Returns:
        1.0 if non-linear, 0.0 if linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    """Read trajectory data from file"""
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
    """Enhanced trajectory dataset with velocity and acceleration features"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, 
                 min_ped=1, delim='\t', include_velocity=True, include_acceleration=True):
        """
        Args:
            data_dir: Directory containing dataset files
            obs_len: Number of observation time steps
            pred_len: Number of prediction time steps
            skip: Frame skip parameter
            threshold: Non-linearity threshold
            min_ped: Minimum number of pedestrians per sequence
            delim: File delimiter
            include_velocity: Include velocity features
            include_acceleration: Include acceleration features
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        
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

        # Convert numpy -> torch tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Convert trajectories to enhanced graphs
        self.V_obs = []
        self.A_obs = []
        self.V_pred = []
        self.A_pred = []

        pbar = tqdm(total=len(self.seq_start_end))
        pbar.set_description(f'Processing {self.data_dir.split("/")[-2]} dataset')

        for ss in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[ss]
            
            # Enhanced graph conversion with velocity and acceleration
            v_, a_ = seq_to_graph(
                self.obs_traj[start:end, :], 
                self.obs_traj_rel[start:end, :],
                include_velocity=self.include_velocity,
                include_acceleration=self.include_acceleration
            )
            self.V_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            
            v_, a_ = seq_to_graph(
                self.pred_traj[start:end, :], 
                self.pred_traj_rel[start:end, :],
                include_velocity=self.include_velocity,
                include_acceleration=self.include_acceleration
            )
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


def data_sampler(V_obs, A_obs, V_tr, A_tr, batch=4, augment=True):
    """
    Enhanced data sampler with augmentation capabilities
    
    Args:
        V_obs: Observed trajectory features
        A_obs: Observed adjacency matrices
        V_tr: Target trajectory features
        A_tr: Target adjacency matrices
        batch: Batch size for sampling
        augment: Whether to apply data augmentation
    """
    if not augment:
        return V_obs, A_obs, V_tr, A_tr
    
    # Create multiple samples through augmentation
    V_obs_batch = []
    A_obs_batch = []
    V_tr_batch = []
    A_tr_batch = []
    
    for _ in range(batch):
        # Apply random transformations
        if torch.rand(1) > 0.5:
            # Random rotation
            angle = torch.rand(1) * 2 * np.pi
            rotation_matrix = torch.tensor([
                [torch.cos(angle), -torch.sin(angle)],
                [torch.sin(angle), torch.cos(angle)]
            ], dtype=V_obs.dtype, device=V_obs.device)
            
            # Apply rotation to trajectory data
            V_obs_rot = V_obs.clone()
            V_tr_rot = V_tr.clone()
            
            # Rotate velocity components
            V_obs_rot = torch.matmul(V_obs_rot.permute(0, 2, 3, 1), rotation_matrix).permute(0, 3, 1, 2)
            V_tr_rot = torch.matmul(V_tr_rot.permute(0, 2, 3, 1), rotation_matrix).permute(0, 3, 1, 2)
            
            V_obs_batch.append(V_obs_rot)
            V_tr_batch.append(V_tr_rot)
        else:
            V_obs_batch.append(V_obs)
            V_tr_batch.append(V_tr)
        
        # Adjacency matrices remain the same (invariant to rotation)
        A_obs_batch.append(A_obs)
        A_tr_batch.append(A_tr)
    
    return (torch.cat(V_obs_batch, dim=0), torch.cat(A_obs_batch, dim=0),
            torch.cat(V_tr_batch, dim=0), torch.cat(A_tr_batch, dim=0))
