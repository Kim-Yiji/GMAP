import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from torch.distributions import multivariate_normal
from copy_dmrgcn.utils.visualizer import generate_statistics_matrices

def create_frame_visualization(V_pred, obs_traj, pred_traj_gt, group_indices=None, 
                            current_frame=0, samples=1000, n_levels=30):
    """Create visualization for a single frame
    
    Args:
        V_pred: [B, T, N, d] tensor of predictions with uncertainty
        obs_traj: [B, T_obs, N, 2] tensor of observed trajectories
        pred_traj_gt: [B, T_pred, N, 2] tensor of ground truth predictions
        group_indices: [B, N] tensor of group assignments
        current_frame: Current frame number (used for trajectory history)
        samples: Number of samples for uncertainty visualization
        n_levels: Number of levels for KDE plot
    """
    # Ensure correct shapes
    assert len(V_pred.shape) == 4, "V_pred should be [B, T, N, d]"
    assert len(obs_traj.shape) == 4, "obs_traj should be [B, T, N, 2]"
    assert len(pred_traj_gt.shape) == 4, "pred_traj_gt should be [B, T, N, 2]"
    
    # Generate ground truth trajectory by concatenating observations and predictions
    V_gt = torch.cat([obs_traj, pred_traj_gt], dim=1)  # [B, T, N, 2]
    V_gt = V_gt[0].permute(1, 0, 2)  # [N, T, 2]
    
    # Get mean and covariance for sampling from predictions
    obs_len = obs_traj.size(1)
    
    # Use last prediction if we're past observation period
    if current_frame >= obs_len:
        pred_frame = current_frame - obs_len
        if pred_frame < V_pred.size(1):
            V_last = V_pred[0, pred_frame]  # Use prediction for current frame [N, d]
        else:
            V_last = V_pred[0, -1]  # Use last prediction if we're past prediction horizon
    else:
        V_last = V_pred[0, 0]  # Use first prediction during observation period
    
    mu, cov = generate_statistics_matrices(V_last)
    
    # Sample trajectories with uncertainty
    V_samples = []
    for n in range(mu.size(0)):
        mv_n = multivariate_normal.MultivariateNormal(mu[n], cov[n])
        V_n = mv_n.sample((samples,))
        V_samples.append(V_n)
    V_samples = torch.stack(V_samples, dim=1)  # [samples, N, 2]
    
    # Use position up to current frame as reference
    obs_len = obs_traj.size(1)
    if current_frame < obs_len:
        last_pos = obs_traj[0, current_frame, :, :2]  # [N, 2]
    else:
        pred_frame = current_frame - obs_len
        if pred_frame < pred_traj_gt.size(1):
            last_pos = pred_traj_gt[0, pred_frame, :, :2]  # [N, 2]
        else:
            last_pos = pred_traj_gt[0, -1, :, :2]  # Use last prediction
    
    # Add samples to current position for absolute coordinates
    V_absl = V_samples + last_pos.unsqueeze(0)  # [samples, N, 2]
    
    # Get full trajectory up to current frame
    if current_frame < obs_traj.size(1):
        # During observation period
        V_gt_full = obs_traj[0, :current_frame+1]  # [T, N, 2]
    else:
        # During prediction period
        pred_frame = current_frame - obs_traj.size(1)
        V_gt_full = torch.cat([
            obs_traj[0],  # All observations
            pred_traj_gt[0, :pred_frame+1]  # Predictions up to current frame
        ], dim=0)
    
    # Prepare data for visualization
    V_absl_temp = V_absl.cpu().numpy()
    V_gt_temp = V_gt_full.permute(1, 0, 2).cpu().numpy()  # [N, T, 2]

    fig = plt.figure(figsize=(10, 7))

    # Generate colors for each group
    if group_indices is not None:
        group_indices_np = group_indices.cpu().numpy()
        unique_groups = np.unique(group_indices_np)
        n_groups = len(unique_groups)
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_groups))
    else:
        colors = [f'C{n}' for n in range(V_gt.size(0))]

    for n in range(V_gt.size(0)):
        if group_indices is not None:
            group_id = group_indices_np[0, n]
            group_idx = np.where(unique_groups == group_id)[0][0]
            color = colors[group_idx]
            label = f'Group {group_id}'
        else:
            color = colors[n]
            label = f'Ped {n}'

        # Plot KDE for predicted trajectories
        samples_n = V_absl_temp[:, n]
        sns.kdeplot(x=samples_n[:, 0], y=samples_n[:, 1],
                   levels=n_levels, fill=True,
                   color=color if isinstance(color, str) else tuple(color),
                   alpha=0.3)
        
        # Plot trajectory up to current frame
        traj_n = V_gt_temp[n, :current_frame+1]  # [T, 2]
        plt.plot(traj_n[:, 0], traj_n[:, 1],
                color=color if isinstance(color, str) else tuple(color),
                linewidth=2, label=label)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tick_params(axis="y", direction="in", pad=-22)
    plt.tick_params(axis="x", direction="in", pad=-15)
    plt.xlim(-14, 36)
    plt.ylim(-9, 26)
    plt.tight_layout()
    
    # Convert figure to image array
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer._renderer)
    plt.close()
    
    return frame

def create_trajectory_video(V_pred, obs_traj, pred_traj_gt, group_indices=None,
                          output_path='trajectory_video.mp4', fps=10):
    """Create video visualization of trajectories
    
    Args:
        V_pred: [B, T, N, d] tensor of predictions with uncertainty
        obs_traj: [B, T_obs, N, 2] tensor of observed trajectories
        pred_traj_gt: [B, T_pred, N, 2] tensor of ground truth predictions
        group_indices: [B, N] tensor of group assignments
        output_path: Path to save the video
        fps: Frames per second for the video
    """
    # Get number of frames
    total_frames = obs_traj.size(1) + pred_traj_gt.size(1)
    
    # Create first frame to get dimensions
    first_frame = create_frame_visualization(V_pred, obs_traj, pred_traj_gt, 
                                          group_indices, current_frame=0)
    height, width, layers = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write first frame
    video.write(cv2.cvtColor(first_frame, cv2.COLOR_RGBA2BGR))
    
    # Create and write remaining frames
    for frame in range(1, total_frames):
        print(f"Processing frame {frame+1}/{total_frames}")
        frame_img = create_frame_visualization(V_pred, obs_traj, pred_traj_gt,
                                            group_indices, current_frame=frame)
        video.write(cv2.cvtColor(frame_img, cv2.COLOR_RGBA2BGR))
    
    # Release video writer
    video.release()
    print(f"Video saved to {output_path}")