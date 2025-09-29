import pickle
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import multivariate_normal
import os
import math
from tqdm import tqdm

from utils import TrajectoryDataset
from model import *
from torch.utils.data import DataLoader
from torch.distributions import multivariate_normal as torch_multivariate_normal

# ETH-UCY-Trajectory-Visualizer의 유틸리티 함수들을 직접 정의
def world2image(coord, H, transpose=False):
    """Convert world coordinates to image coordinates."""
    assert coord.shape[-1] == 2
    assert H.shape == (3, 3)
    assert type(coord) == type(H)

    shape = coord.shape
    coord = coord.reshape(-1, 2)

    if isinstance(coord, np.ndarray):
        x, y = coord[..., 0], coord[..., 1]
        image = (np.linalg.inv(H) @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
        image = image / image[..., [2]]
        image = image[..., :2]
    
    elif isinstance(coord, torch.Tensor):
        x, y = coord[..., 0], coord[..., 1]
        image = (torch.linalg.inv(H) @ torch.stack([x, y, torch.ones_like(x)], dim=-1).T).T
        image = image / image[..., [2]]
        image = image[..., :2]

    else:
        raise NotImplementedError
    
    return image.reshape(shape)

def image2world(coord, H):
    """Convert image coordinates to world coordinates."""
    assert coord.shape[-1] == 2
    assert H.shape == (3, 3)
    assert type(coord) == type(H)

    shape = coord.shape
    coord = coord.reshape(-1, 2)

    if isinstance(coord, np.ndarray):
        x, y = coord[..., 0], coord[..., 1]
        world = (H @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
        world = world / world[..., [2]]
        world = world[..., :2]
    
    elif isinstance(coord, torch.Tensor):
        x, y = coord[..., 0], coord[..., 1]
        world = (H @ torch.stack([x, y, torch.ones_like(x)], dim=-1).T).T
        world = world / world[..., [2]]
        world = world[..., :2]
        
    else:
        raise NotImplementedError
    
    return world.reshape(shape)

# Video dataset class
class VideoTrajectoryDataset:
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, min_ped=1, delim='\t'):
        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.cur_frame_no = 0

        all_files = sorted(os.listdir(self.data_dir + '/test/'))
        all_files = [os.path.join(self.data_dir + '/test/', _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        frame_list = []

        for path in all_files:
            data = self.read_file(path, delim)
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
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    num_peds_considered += 1
                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    frame_list.append(frames[idx])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        self.obs_traj = seq_list[:, :, :self.obs_len]
        self.pred_traj = seq_list[:, :, self.obs_len:]
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.frame_list = np.array(frame_list, dtype=np.int32)

        with open(self.data_dir + '/H.txt', 'r') as h_mat_file:
            h_mat = h_mat_file.read()
        h_mat = [x.split() for x in h_mat.split('\n')][:3][:3]
        self.h_mat = np.array(h_mat).astype(np.float32)

        self.cap = cv2.VideoCapture(self.data_dir + '/video.avi')
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_shape = [width, height, 3]

    def read_file(self, _path, delim):
        delim = delim if delim else self.delim
        data = []
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

    def get_image_from_frame(self, frame_no):
        # Reset video capture to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return np.ones(self.video_shape, dtype=np.uint8)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        frame = self.frame_list[index]  # Remove offset to process all frames
        out = [self.obs_traj[start:end, :], self.pred_traj[start:end, :],
               self.frame_list[index], self.get_image_from_frame(frame)]
        return out

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='social-dmrgcn-hotel-experiment_tp4_de80', help='Personal tag for the model')
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples for probability map')
parser.add_argument('--dataset', default='hotel', help='Dataset name')
parser.add_argument('--output_dir', default='./video_visualization_output/', help='Output directory for videos')
parser.add_argument('--frame_start', type=int, default=0, help='Starting frame for visualization')
parser.add_argument('--frame_end', type=int, default=10, help='Ending frame for visualization')
test_args = parser.parse_args()

# Get arguments for training
checkpoint_dir = './checkpoints/' + test_args.tag + '/'
args_path = checkpoint_dir + '/args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)

dataset_path = './datasets/' + test_args.dataset + '/'
model_path = checkpoint_dir + test_args.dataset + '_best.pth'
KSTEPS = test_args.n_samples

# Data preparation
test_dataset = TrajectoryDataset(dataset_path + 'test/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Video dataset preparation
video_dataset_path = './ETH-UCY-Trajectory-Visualizer/datasets_visualize/' + test_args.dataset + '/'
video_dataset = VideoTrajectoryDataset(video_dataset_path, obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)

# Model preparation
model = social_dmrgcn(n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn,
                      output_feat=args.output_size, kernel_size=args.kernel_size,
                      seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len)
# Use CPU instead of CUDA
device = torch.device('cpu')
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# Create output directory
os.makedirs(test_args.output_dir, exist_ok=True)

def generate_probability_map(predictions, image_shape, homography_matrix, sigma=5.0):
    """
    Generate probability map from trajectory predictions
    
    Args:
        predictions: numpy array of shape (n_samples, n_peds, pred_len, 2) - world coordinates
        image_shape: tuple (height, width)
        homography_matrix: 3x3 homography matrix
        sigma: standard deviation for Gaussian kernel
    
    Returns:
        probability_map: numpy array of shape (height, width)
    """
    height, width = image_shape
    prob_map = np.zeros((height, width))
    
    n_samples, n_peds, pred_len, _ = predictions.shape
    
    for ped_idx in range(n_peds):
        for t in range(pred_len):
            # Get all sample positions for this pedestrian at this time step
            positions = predictions[:, ped_idx, t, :]  # (n_samples, 2)
            
            # Convert world coordinates to image coordinates
            image_positions = world2image(positions, homography_matrix)
            
            # Filter out positions outside image bounds
            valid_mask = (image_positions[:, 0] >= 0) & (image_positions[:, 0] < width) & \
                        (image_positions[:, 1] >= 0) & (image_positions[:, 1] < height)
            
            if np.sum(valid_mask) == 0:
                continue
                
            valid_positions = image_positions[valid_mask]
            
            # Create Gaussian kernel for each valid position
            for pos in valid_positions:
                x, y = int(pos[0]), int(pos[1])
                
                # Create Gaussian kernel around this position
                kernel_size = int(3 * sigma)
                x_min = max(0, x - kernel_size)
                x_max = min(width, x + kernel_size + 1)
                y_min = max(0, y - kernel_size)
                y_max = min(height, y + kernel_size + 1)
                
                # Generate Gaussian kernel
                x_coords = np.arange(x_min, x_max)
                y_coords = np.arange(y_min, y_max)
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Calculate Gaussian values
                gaussian_values = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
                
                # Add to probability map
                prob_map[y_min:y_max, x_min:x_max] += gaussian_values
    
    # Normalize probability map
    if np.max(prob_map) > 0:
        prob_map = prob_map / np.max(prob_map)
    
    return prob_map

def create_colormap():
    """Create a custom colormap for probability visualization"""
    colors = ['black', 'blue', 'cyan', 'yellow', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('probability', colors, N=n_bins)
    return cmap

def visualize_frame_with_probability_map(frame_idx, obs_traj, pred_traj_gt, V_pred, image, homography_matrix):
    """
    Visualize a single frame with probability map overlay
    
    Args:
        frame_idx: frame index
        obs_traj: observed trajectory (world coordinates)
        pred_traj_gt: ground truth prediction trajectory (world coordinates)
        V_pred: model predictions (relative coordinates)
        image: video frame image
        homography_matrix: homography matrix for coordinate transformation
    
    Returns:
        visualized_frame: numpy array of the visualized frame
    """
    # Generate trajectory samples
    mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
    mv_normal = torch_multivariate_normal.MultivariateNormal(mu, cov)
    V_pred_sample = mv_normal.sample((KSTEPS,))
    
    # Convert relative trajectories to absolute trajectories
    V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
    V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)
    
    V_absl = []
    for t in range(V_pred_sample.size(1)):
        V_absl.append(V_pred_sample[:, 0:t + 1, :, :].sum(dim=1, keepdim=True) + V_obs_traj[-1, :, :])
    V_absl = torch.cat(V_absl, dim=1)
    
    # Convert to numpy and world coordinates
    V_absl_np = V_absl.cpu().numpy()  # (n_samples, pred_len, n_peds, 2)
    V_absl_np = V_absl_np.transpose(0, 2, 1, 3)  # (n_samples, n_peds, pred_len, 2)
    
    # Generate probability map
    prob_map = generate_probability_map(V_absl_np, image.shape[:2], homography_matrix)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display original image
    ax.imshow(image, origin='lower')
    
    # Overlay probability map with transparency
    cmap = create_colormap()
    prob_overlay = ax.imshow(prob_map, cmap=cmap, alpha=0.8, origin='lower')
    
    # Draw ground truth trajectories
    obs_traj_np = obs_traj.squeeze().cpu().numpy()  # (n_peds, obs_len, 2)
    pred_traj_gt_np = pred_traj_gt.squeeze().cpu().numpy()  # (n_peds, pred_len, 2)
    
    for ped_idx in range(obs_traj_np.shape[0]):
        # Convert to image coordinates - transpose to get (time, 2) format
        obs_traj_ped = obs_traj_np[ped_idx].T  # (8, 2)
        pred_traj_gt_ped = pred_traj_gt_np[ped_idx].T  # (12, 2)
        
        obs_traj_img = world2image(obs_traj_ped, homography_matrix)
        pred_traj_gt_img = world2image(pred_traj_gt_ped, homography_matrix)
        
        # Draw observed trajectory (blue)
        ax.plot(obs_traj_img[:, 0], obs_traj_img[:, 1], 'b-', linewidth=3, label='Observed' if ped_idx == 0 else "")
        
        # Draw ground truth prediction trajectory (red)
        ax.plot(pred_traj_gt_img[:, 0], pred_traj_gt_img[:, 1], 'r-', linewidth=3, label='Ground Truth' if ped_idx == 0 else "")
    
    # Set axis properties
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.invert_yaxis()
    ax.set_title(f'Frame {frame_idx} - Probability Map Overlay')
    ax.legend()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    vis_frame = np.asarray(buf)
    vis_frame = vis_frame[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return vis_frame

def test_with_video_visualization():
    """Test the model and create video visualization with probability maps"""
    model.eval()
    
    # Create video writer
    output_video_path = os.path.join(test_args.output_dir, f'{test_args.dataset}_probability_map.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Get video properties from the first frame
    first_frame_data = video_dataset[0]
    sample_image = first_frame_data[3]
    height, width = sample_image.shape[:2]
    
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))
    
    print(f"Processing frames {test_args.frame_start} to {test_args.frame_end}")
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing frames")):
        # Remove frame limits to process all frames
        # if batch_idx < test_args.frame_start:
        #     continue
        # if batch_idx >= test_args.frame_end:
        #     break
            
        # Get model predictions
        V_obs, A_obs, V_tr, A_tr = [tensor.to(device) for tensor in batch[-4:]]
        obs_traj, pred_traj_gt = [tensor.to(device) for tensor in batch[:2]]
        
        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)
        
        # Get corresponding video frame
        video_frame_data = video_dataset[batch_idx]
        video_image = video_frame_data[3]
        homography_matrix = video_dataset.h_mat
        
        # Create visualization
        vis_frame = visualize_frame_with_probability_map(
            batch_idx, obs_traj, pred_traj_gt, V_pred, video_image, homography_matrix
        )
        
        # Resize to match video dimensions
        vis_frame_resized = cv2.resize(vis_frame, (width, height))
        
        # Convert RGB to BGR for OpenCV
        vis_frame_bgr = cv2.cvtColor(vis_frame_resized, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        video_writer.write(vis_frame_bgr)
        
        # Save individual frame as image
        frame_path = os.path.join(test_args.output_dir, f'frame_{batch_idx:04d}.png')
        cv2.imwrite(frame_path, vis_frame_bgr)
    
    video_writer.release()
    print(f"Video saved to: {output_video_path}")
    print(f"Individual frames saved to: {test_args.output_dir}")

def main():
    test_with_video_visualization()

if __name__ == "__main__":
    main()
