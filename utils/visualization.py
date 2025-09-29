import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_static_visualization(obs_traj, pred_traj, group_indices, 
                              title="Trajectory Visualization"):
    """Create a static visualization of trajectories with group information
    
    Args:
        obs_traj: Observed trajectories [B, T, N, 2]
        pred_traj: Predicted trajectories [B, T, N, 2]
        group_indices: Group assignments [B, N]
        title: Plot title
        
    Returns:
        fig: Matplotlib figure object
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy for easier handling
    obs_traj = obs_traj.squeeze().cpu().numpy()  # [T, N, 2]
    pred_traj = pred_traj.squeeze().cpu().numpy()  # [T, N, 2]
    group_indices = group_indices.squeeze().cpu().numpy()  # [N]
    
    # Get unique group IDs for coloring
    unique_groups = np.unique(group_indices)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_groups)))
    group_colors = {g: c for g, c in zip(unique_groups, colors)}
    
    # Plot trajectories for each pedestrian
    for n in range(obs_traj.shape[1]):
        group_id = group_indices[n]
        color = group_colors[group_id]
        
        # Observed trajectory
        plt.plot(obs_traj[:, n, 0], obs_traj[:, n, 1], 
                '-o', color=color, alpha=0.5, label=f'Group {group_id}')
        
        # Predicted trajectory
        plt.plot(pred_traj[:, n, 0], pred_traj[:, n, 1], 
                '--', color=color, alpha=0.5)
        
        # Current position
        plt.plot(obs_traj[-1, n, 0], obs_traj[-1, n, 1], 
                'o', color=color, markersize=10)
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Convert plot to image array
    plt.tight_layout()
    fig = plt.gcf()
    fig.canvas.draw()
    
    # Convert to image array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return img

def create_trajectory_video(obs_traj, pred_traj, group_indices, output_path, fps=2.5):
    """Create a video visualization of trajectories with group information
    
    Args:
        obs_traj: Observed trajectories [B, T, N, 2]
        pred_traj: Predicted trajectories [B, T, N, 2]
        group_indices: Group assignments [B, N]
        output_path: Path to save the video
        fps: Frames per second
    """
    obs_traj = obs_traj.squeeze().cpu().numpy()  # [T, N, 2]
    pred_traj = pred_traj.squeeze().cpu().numpy()  # [T, N, 2]
    group_indices = group_indices.squeeze().cpu().numpy()  # [N]
    
    T_obs = obs_traj.shape[0]
    T_pred = pred_traj.shape[0]
    
    # Get unique group IDs for coloring
    unique_groups = np.unique(group_indices)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_groups)))
    group_colors = {g: c for g, c in zip(unique_groups, colors)}
    
    # Set up video writer
    fig = plt.figure(figsize=(12, 8))
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Get axis limits
    min_x = min(obs_traj[..., 0].min(), pred_traj[..., 0].min()) - 1
    max_x = max(obs_traj[..., 0].max(), pred_traj[..., 0].max()) + 1
    min_y = min(obs_traj[..., 1].min(), pred_traj[..., 1].min()) - 1
    max_y = max(obs_traj[..., 1].max(), pred_traj[..., 1].max()) + 1
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    # Convert figure to image
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Initialize video writer
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Create frames for observed trajectory
    for t in range(T_obs):
        plt.clf()
        plt.grid(True, alpha=0.3)
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        
        for n in range(obs_traj.shape[1]):
            group_id = group_indices[n]
            color = group_colors[group_id]
            
            # Plot trajectory up to current time
            plt.plot(obs_traj[:t+1, n, 0], obs_traj[:t+1, n, 1], 
                    '-o', color=color, alpha=0.5)
            
            # Plot current position
            if t == T_obs - 1:
                plt.plot(obs_traj[t, n, 0], obs_traj[t, n, 1], 
                        'o', color=color, markersize=10)
        
        plt.title(f'Observation t={t+1}/{T_obs}')
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    
    # Create frames for predicted trajectory
    for t in range(T_pred):
        plt.clf()
        plt.grid(True, alpha=0.3)
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        
        for n in range(pred_traj.shape[1]):
            group_id = group_indices[n]
            color = group_colors[group_id]
            
            # Plot observed trajectory
            plt.plot(obs_traj[:, n, 0], obs_traj[:, n, 1], 
                    '-o', color=color, alpha=0.5)
            
            # Plot prediction up to current time
            plt.plot(pred_traj[:t+1, n, 0], pred_traj[:t+1, n, 1], 
                    '--', color=color, alpha=0.5)
            
            # Plot current position
            plt.plot(pred_traj[t, n, 0], pred_traj[t, n, 1], 
                    'o', color=color, markersize=10)
        
        plt.title(f'Prediction t={t+1}/{T_pred}')
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    
    writer.release()
    plt.close()