import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions import multivariate_normal

def generate_statistics_matrices(V_pred):
    """Generate mean and covariance matrices for trajectory prediction
    
    Args:
        V_pred: [N, d] tensor of trajectory predictions
        where d includes position (2) and uncertainty parameters (3)
        
    Returns:
        mu: (N, 2) mean of predicted positions
        cov: (N, 2, 2) covariance matrix of predicted positions
    """
    assert len(V_pred.shape) == 2, "Expected [N, d] tensor"
    N, d = V_pred.shape
    assert d >= 5, "Expected at least 5 dimensions (2 for position, 3 for uncertainty)"
    
    # Extract mean positions (first 2 dimensions)
    mu = V_pred[:, :2]  # [N, 2]
    
    # Generate covariance matrices from uncertainty parameters
    var = torch.exp(V_pred[:, 2])  # [N] - variance
    corr = torch.tanh(V_pred[:, 3])  # [N] - correlation in [-1, 1]
    ratio = torch.exp(V_pred[:, 4])  # [N] - axis ratio
    
    # Construct 2x2 covariance matrices for each pedestrian
    cov = torch.zeros(N, 2, 2, device=V_pred.device)
    for n in range(N):
        # Principal variances
        s1 = var[n]
        s2 = var[n] * ratio[n]
        r = corr[n]
        
        # Covariance matrix
        cov[n] = torch.tensor([[s1, r * torch.sqrt(s1 * s2)],
                             [r * torch.sqrt(s1 * s2), s2]], device=V_pred.device)
    
    # Add small diagonal term for numerical stability
    cov = cov + torch.eye(2, device=cov.device).unsqueeze(0) * 1e-5
    
    return mu, cov


def figure_to_array(fig):
    r"""Convert plt.figure to RGBA numpy array. shape: height, width, layer"""

    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def data_visualizer(V_pred, obs_traj, pred_traj_gt, group_indices=None, samples=1000, n_levels=30):
    """Visualize trajectories with group information
    
    Args:
        V_pred: [B, T, N, d] tensor of predictions with uncertainty
        obs_traj: [B, T_obs, N, 2] tensor of observed trajectories
        pred_traj_gt: [B, T_pred, N, 2] tensor of ground truth predictions
        group_indices: [B, N] tensor of group assignments
    """
    # Ensure correct shapes
    assert len(V_pred.shape) == 4, "V_pred should be [B, T, N, d]"
    assert len(obs_traj.shape) == 4, "obs_traj should be [B, T, N, 2]"
    assert len(pred_traj_gt.shape) == 4, "pred_traj_gt should be [B, T, N, 2]"
    
    # Generate ground truth trajectory by concatenating observations and predictions
    V_gt = torch.cat([obs_traj, pred_traj_gt], dim=1)  # [B, T, N, 2]
    V_gt = V_gt[0].permute(1, 0, 2)  # [N, T, 2] - permute for per-agent processing
    
    # Get mean and covariance for sampling from predictions
    assert len(V_pred.shape) == 4, "V_pred should be [B, T, N, d]"
    V_last = V_pred[0, -1]  # Use last timestep predictions [N, d]
    mu, cov = generate_statistics_matrices(V_last)  # (N, 2), (N, 2, 2)
    
    # Sample trajectories with uncertainty
    print(f"mu shape: {mu.shape}")  # Debug print
    
    V_samples = []
    for n in range(mu.size(0)):  # For each pedestrian
        mv_n = multivariate_normal.MultivariateNormal(mu[n], cov[n])
        V_n = mv_n.sample((samples,))  # [samples, 2]
        V_samples.append(V_n)
    V_samples = torch.stack(V_samples, dim=1)  # [samples, N, 2]
    
    print(f"V_samples shape: {V_samples.shape}")  # Debug print
    
    # Use last observed position as reference
    obs_length = obs_traj.size(1)
    last_obs_pos = obs_traj[0, obs_length-1, :, :2]  # [N, 2]
    
    # Add samples to last observed position for absolute coordinates
    V_absl = V_samples + last_obs_pos.unsqueeze(0)  # [samples, N, 2]
    
    # Prepare ground truth trajectory (convert to absolute coordinates)
    V_gt_full = torch.cat([obs_traj[0, :, :, :2], pred_traj_gt[0, :, :, :2]], dim=0)  # [T, N, 2]

    # Prepare data for visualization
    V_absl_temp = V_absl.cpu().numpy()       # [samples, N, 2]
    V_gt_temp = V_gt_full.permute(1, 0, 2).cpu().numpy()  # [N, T, 2]

    fig = plt.figure(figsize=(10, 7))

    # Generate colors for each group
    if group_indices is not None:
        # Convert to numpy while preserving the shape
        group_indices_np = group_indices.cpu().numpy()  # Keep as 2D array [B, N]
        unique_groups = np.unique(group_indices_np)
        n_groups = len(unique_groups)
        # Create colormap with distinct colors for each group
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_groups))
    else:
        colors = [f'C{n}' for n in range(V_gt.size(0))]

    for n in range(V_gt.size(0)):  # For each pedestrian
        # Get color based on group
        if group_indices is not None:
            # Access group index for current pedestrian
            group_id = group_indices_np[0, n]  # Get from first batch, nth pedestrian
            group_idx = np.where(unique_groups == group_id)[0][0]
            color = colors[group_idx]
            label = f'Group {group_id}'
        else:
            color = colors[n]
            label = f'Ped {n}'

        # Plot KDE for predicted trajectories
        samples_n = V_absl_temp[:, n]  # [samples, 2]
        sns.kdeplot(x=samples_n[:, 0], y=samples_n[:, 1],
                   levels=n_levels, fill=True,
                   color=color if isinstance(color, str) else tuple(color),
                   alpha=0.3)
        
        # Plot ground truth trajectory
        traj_n = V_gt_temp[n]  # [T, 2]
        
        # Plot observation (solid line) and prediction (dashed line)
        obs_len = obs_traj.size(1)
        plt.plot(traj_n[:obs_len, 0], traj_n[:obs_len, 1],
                color=color if isinstance(color, str) else tuple(color),
                linewidth=2, label=f"{label} (obs)")
        plt.plot(traj_n[obs_len:, 0], traj_n[obs_len:, 1],
                linestyle='--', color=color if isinstance(color, str) else tuple(color),
                linewidth=1, label=f"{label} (pred)")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tick_params(axis="y", direction="in", pad=-22)
    plt.tick_params(axis="x", direction="in", pad=-15)
    plt.xlim(-14, 36)
    plt.ylim(-9, 26)
    plt.tight_layout()

    plt.close()

    return figure_to_array(fig)


def visualize_scene(data_loader, model, frame_id, group_indices=None):
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx == frame_id:
            V_obs, A_obs, V_tr, A_tr = [tensor.cuda() for tensor in batch[-4:]]
            obs_traj, pred_traj_gt = [tensor.cuda() for tensor in batch[:2]]

            V_obs_ = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_, A_obs)
            V_pred = V_pred.permute(0, 2, 3, 1)

            V_pred = V_pred.squeeze()
            V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
            V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

            # Visualize trajectories with group information
            fig_img = data_visualizer(V_pred.unsqueeze(dim=0), obs_traj, pred_traj_gt,
                                    group_indices=group_indices)

            plt.imshow(fig_img[:, :, :3])
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            break
