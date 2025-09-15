import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import seaborn as sns
from matplotlib.animation import FuncAnimation
import io
from PIL import Image


def trajectory_visualizer(pred_traj, obs_traj, gt_traj, group_indices=None, 
                         samples=20, show_uncertainty=True, save_path=None):
    """
    Visualize trajectory predictions with group information
    
    Args:
        pred_traj: Predicted trajectories [samples, time, ped, 2] or [time, ped, 2]
        obs_traj: Observed trajectories [time, ped, 2]
        gt_traj: Ground truth trajectories [time, ped, 2]
        group_indices: Group membership indices [ped]
        samples: Number of samples to visualize
        show_uncertainty: Whether to show prediction uncertainty
        save_path: Path to save the figure
    
    Returns:
        Figure as numpy array for tensorboard logging
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert tensors to numpy
    if torch.is_tensor(pred_traj):
        pred_traj = pred_traj.detach().cpu().numpy()
    if torch.is_tensor(obs_traj):
        obs_traj = obs_traj.detach().cpu().numpy()
    if torch.is_tensor(gt_traj):
        gt_traj = gt_traj.detach().cpu().numpy()
    if torch.is_tensor(group_indices):
        group_indices = group_indices.detach().cpu().numpy()
    
    # Handle different prediction formats
    if len(pred_traj.shape) == 4:  # [samples, time, ped, 2]
        pred_samples = pred_traj[:samples]
        pred_mean = np.mean(pred_samples, axis=0)
        pred_std = np.std(pred_samples, axis=0)
    else:  # [time, ped, 2]
        pred_mean = pred_traj
        pred_samples = None
        pred_std = None
    
    num_peds = obs_traj.shape[1]
    
    # Define colors for different groups
    if group_indices is not None:
        unique_groups = np.unique(group_indices)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
        group_colors = {group_id: colors[i] for i, group_id in enumerate(unique_groups)}
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_peds))
        group_colors = {i: colors[i] for i in range(num_peds)}
    
    # Plot trajectories for each pedestrian
    for ped in range(num_peds):
        # Determine color based on group
        if group_indices is not None:
            color = group_colors[group_indices[ped]]
        else:
            color = group_colors[ped]
        
        # Plot observed trajectory
        ax.plot(obs_traj[:, ped, 0], obs_traj[:, ped, 1], 
                'o-', color=color, linewidth=2, markersize=4, 
                label=f'Ped {ped} (Obs)' if ped == 0 else "", alpha=0.8)
        
        # Plot ground truth trajectory
        ax.plot(gt_traj[:, ped, 0], gt_traj[:, ped, 1], 
                's--', color=color, linewidth=2, markersize=4,
                label=f'Ground Truth' if ped == 0 else "", alpha=0.8)
        
        # Plot predicted trajectory
        ax.plot(pred_mean[:, ped, 0], pred_mean[:, ped, 1], 
                '^-', color=color, linewidth=2, markersize=4,
                label=f'Prediction' if ped == 0 else "", alpha=0.8)
        
        # Show uncertainty if available
        if show_uncertainty and pred_std is not None:
            for t in range(pred_mean.shape[0]):
                circle = patches.Ellipse(
                    (pred_mean[t, ped, 0], pred_mean[t, ped, 1]),
                    pred_std[t, ped, 0] * 2, pred_std[t, ped, 1] * 2,
                    alpha=0.2, color=color)
                ax.add_patch(circle)
        
        # Show all prediction samples
        if pred_samples is not None and show_uncertainty:
            for sample in pred_samples[:min(5, samples)]:  # Show only first 5 samples
                ax.plot(sample[:, ped, 0], sample[:, ped, 1], 
                        '-', color=color, alpha=0.1, linewidth=1)
        
        # Mark start and end points
        ax.plot(obs_traj[0, ped, 0], obs_traj[0, ped, 1], 
                'o', color='green', markersize=8, markeredgecolor='black')
        ax.plot(gt_traj[-1, ped, 0], gt_traj[-1, ped, 1], 
                's', color='red', markersize=8, markeredgecolor='black')
    
    # Add group information
    if group_indices is not None:
        # Add legend for groups
        for group_id, color in group_colors.items():
            group_peds = np.where(group_indices == group_id)[0]
            ax.scatter([], [], c=color, s=100, alpha=0.7, 
                      label=f'Group {group_id} (Peds: {list(group_peds)})')
    
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.set_title('Trajectory Prediction Visualization')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array for tensorboard
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img_array


def plot_group_analysis(group_indices, positions, velocities=None, title="Group Analysis"):
    """
    Visualize group formation and dynamics
    
    Args:
        group_indices: Group membership indices
        positions: Pedestrian positions [time, ped, 2]
        velocities: Pedestrian velocities [time, ped, 2] (optional)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if torch.is_tensor(group_indices):
        group_indices = group_indices.detach().cpu().numpy()
    if torch.is_tensor(positions):
        positions = positions.detach().cpu().numpy()
    if velocities is not None and torch.is_tensor(velocities):
        velocities = velocities.detach().cpu().numpy()
    
    unique_groups = np.unique(group_indices)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
    
    # Plot 1: Group positions over time
    ax1 = axes[0]
    for i, group_id in enumerate(unique_groups):
        group_mask = group_indices == group_id
        group_positions = positions[:, group_mask, :]
        
        # Plot trajectories for this group
        for ped in range(group_positions.shape[1]):
            ax1.plot(group_positions[:, ped, 0], group_positions[:, ped, 1], 
                    'o-', color=colors[i], alpha=0.7, 
                    label=f'Group {group_id}' if ped == 0 else "")
    
    ax1.set_xlabel('X coordinate (m)')
    ax1.set_ylabel('Y coordinate (m)')
    ax1.set_title('Group Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Group statistics
    ax2 = axes[1]
    group_sizes = []
    group_spreads = []
    
    for group_id in unique_groups:
        group_mask = group_indices == group_id
        group_size = np.sum(group_mask)
        group_positions_final = positions[-1, group_mask, :]
        
        # Calculate group spread (average distance from centroid)
        centroid = np.mean(group_positions_final, axis=0)
        distances = np.linalg.norm(group_positions_final - centroid, axis=1)
        group_spread = np.mean(distances)
        
        group_sizes.append(group_size)
        group_spreads.append(group_spread)
    
    bars = ax2.bar(range(len(unique_groups)), group_sizes, 
                   color=colors, alpha=0.7)
    ax2.set_xlabel('Group ID')
    ax2.set_ylabel('Group Size')
    ax2.set_title('Group Sizes')
    ax2.set_xticks(range(len(unique_groups)))
    ax2.set_xticklabels([f'G{gid}' for gid in unique_groups])
    
    # Add group spread as text annotations
    for i, (size, spread) in enumerate(zip(group_sizes, group_spreads)):
        ax2.text(i, size + 0.1, f'Spread: {spread:.2f}m', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img_array


def plot_metrics_over_time(metrics_history, save_path=None):
    """
    Plot training metrics over time
    
    Args:
        metrics_history: Dictionary of metric lists over epochs
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metric_names = ['train_loss', 'val_loss', 'ADE', 'FDE']
    
    for i, metric_name in enumerate(metric_names):
        if metric_name in metrics_history:
            epochs = range(len(metrics_history[metric_name]))
            axes[i].plot(epochs, metrics_history[metric_name], 'b-', linewidth=2)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric_name.upper())
            axes[i].set_title(f'{metric_name.upper()} over Training')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img_array


def create_trajectory_animation(pred_samples, obs_traj, gt_traj, group_indices=None, 
                               save_path=None, fps=5):
    """
    Create animated visualization of trajectory prediction
    
    Args:
        pred_samples: Predicted trajectory samples [samples, time, ped, 2]
        obs_traj: Observed trajectories [time, ped, 2]
        gt_traj: Ground truth trajectories [time, ped, 2]
        group_indices: Group membership indices
        save_path: Path to save animation
        fps: Frames per second
    """
    # Convert tensors to numpy
    if torch.is_tensor(pred_samples):
        pred_samples = pred_samples.detach().cpu().numpy()
    if torch.is_tensor(obs_traj):
        obs_traj = obs_traj.detach().cpu().numpy()
    if torch.is_tensor(gt_traj):
        gt_traj = gt_traj.detach().cpu().numpy()
    if torch.is_tensor(group_indices):
        group_indices = group_indices.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine plot bounds
    all_positions = np.concatenate([obs_traj, gt_traj], axis=0)
    x_min, x_max = all_positions[:, :, 0].min(), all_positions[:, :, 0].max()
    y_min, y_max = all_positions[:, :, 1].min(), all_positions[:, :, 1].max()
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    
    # Setup colors
    num_peds = obs_traj.shape[1]
    if group_indices is not None:
        unique_groups = np.unique(group_indices)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
        ped_colors = [colors[group_indices[ped]] for ped in range(num_peds)]
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, num_peds))
        ped_colors = colors
    
    # Initialize plot elements
    obs_lines = []
    gt_lines = []
    pred_lines = []
    current_positions = []
    
    for ped in range(num_peds):
        obs_line, = ax.plot([], [], 'o-', color=ped_colors[ped], 
                           linewidth=2, alpha=0.8, label=f'Obs {ped}')
        gt_line, = ax.plot([], [], 's-', color=ped_colors[ped], 
                          linewidth=2, alpha=0.8, linestyle='--')
        pred_line, = ax.plot([], [], '^-', color=ped_colors[ped], 
                            linewidth=2, alpha=0.8)
        current_pos, = ax.plot([], [], 'o', color=ped_colors[ped], 
                              markersize=10, markeredgecolor='black')
        
        obs_lines.append(obs_line)
        gt_lines.append(gt_line)
        pred_lines.append(pred_line)
        current_positions.append(current_pos)
    
    def animate(frame):
        obs_end = obs_traj.shape[0]
        total_frames = obs_end + gt_traj.shape[0]
        
        for ped in range(num_peds):
            if frame < obs_end:
                # Show observed trajectory up to current frame
                obs_lines[ped].set_data(obs_traj[:frame+1, ped, 0], 
                                       obs_traj[:frame+1, ped, 1])
                current_positions[ped].set_data([obs_traj[frame, ped, 0]], 
                                               [obs_traj[frame, ped, 1]])
            else:
                # Show full observed trajectory and prediction/ground truth
                obs_lines[ped].set_data(obs_traj[:, ped, 0], obs_traj[:, ped, 1])
                
                pred_frame = frame - obs_end
                if pred_frame < gt_traj.shape[0]:
                    gt_lines[ped].set_data(gt_traj[:pred_frame+1, ped, 0], 
                                          gt_traj[:pred_frame+1, ped, 1])
                    
                    # Show prediction mean
                    pred_mean = np.mean(pred_samples, axis=0)
                    pred_lines[ped].set_data(pred_mean[:pred_frame+1, ped, 0], 
                                            pred_mean[:pred_frame+1, ped, 1])
                    
                    current_positions[ped].set_data([gt_traj[pred_frame, ped, 0]], 
                                                   [gt_traj[pred_frame, ped, 1]])
        
        ax.set_title(f'Frame {frame}/{total_frames-1}')
        return obs_lines + gt_lines + pred_lines + current_positions
    
    total_frames = obs_traj.shape[0] + gt_traj.shape[0]
    anim = FuncAnimation(fig, animate, frames=total_frames, 
                        interval=1000//fps, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
    
    plt.close(fig)
    return anim


def visualize_attention_weights(attention_weights, group_indices=None, save_path=None):
    """
    Visualize attention weights between pedestrians
    
    Args:
        attention_weights: Attention weight matrix [ped, ped]
        group_indices: Group membership indices
        save_path: Path to save the figure
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.detach().cpu().numpy()
    if torch.is_tensor(group_indices):
        group_indices = group_indices.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='Blues', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set labels
    num_peds = attention_weights.shape[0]
    ax.set_xticks(range(num_peds))
    ax.set_yticks(range(num_peds))
    ax.set_xticklabels([f'Ped {i}' for i in range(num_peds)])
    ax.set_yticklabels([f'Ped {i}' for i in range(num_peds)])
    
    # Add group information if available
    if group_indices is not None:
        unique_groups = np.unique(group_indices)
        
        # Add group boundaries
        for group_id in unique_groups:
            group_peds = np.where(group_indices == group_id)[0]
            if len(group_peds) > 1:
                min_ped, max_ped = group_peds.min(), group_peds.max()
                # Draw rectangle around group
                rect = patches.Rectangle((min_ped-0.5, min_ped-0.5), 
                                       max_ped-min_ped+1, max_ped-min_ped+1,
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none')
                ax.add_patch(rect)
    
    ax.set_xlabel('Target Pedestrian')
    ax.set_ylabel('Source Pedestrian')
    ax.set_title('Attention Weights Between Pedestrians')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img_array
