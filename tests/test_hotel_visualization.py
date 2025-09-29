import os
import sys
import torch
import numpy as np

# Add the root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from datasets.dataloader import read_file
from model.gpgraph_adapter import GroupAssignment
from copy_dmrgcn.utils.video_visualizer import create_trajectory_video
from copy_dmrgcn.utils.visualizer import data_visualizer

def load_hotel_trajectories(data_path, obs_len=8, pred_len=8):
    """Load and process hotel trajectory data"""
    # Read data file
    data = read_file(os.path.join(data_path, 'biwi_hotel.txt'), delim='\t')
    
    # Get unique frames and pedestrians
    frames = np.unique(data[:, 0])
    frame_data = []
    
    # Group data by frames
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :])
    
    # Process first sequence for visualization
    seq_data = frame_data[:obs_len + pred_len]
    
    # Get unique pedestrians in sequence
    peds = np.unique(np.concatenate([d[:, 1] for d in seq_data]))
    num_peds = len(peds)
    
    # Initialize trajectory tensors
    obs_traj = torch.zeros((1, obs_len, num_peds, 2))  # [B, T, N, 2]
    pred_traj = torch.zeros((1, pred_len, num_peds, 2))  # [B, T, N, 2]
    
    # Fill trajectories
    for t, frame_data in enumerate(seq_data):
        for p_idx, ped_id in enumerate(peds):
            ped_data = frame_data[frame_data[:, 1] == ped_id]
            if len(ped_data) > 0:
                if t < obs_len:
                    obs_traj[0, t, p_idx] = torch.tensor(ped_data[0, 2:4])
                else:
                    pred_traj[0, t-obs_len, p_idx] = torch.tensor(ped_data[0, 2:4])
    
    # Calculate relative trajectories
    obs_traj_rel = torch.zeros_like(obs_traj)
    obs_traj_rel[:, 1:] = obs_traj[:, 1:] - obs_traj[:, :-1]
    
    pred_traj_rel = torch.zeros_like(pred_traj)
    pred_traj_rel[:, 1:] = pred_traj[:, 1:] - pred_traj[:, :-1]
    
    return obs_traj, pred_traj, obs_traj_rel, pred_traj_rel

def test_hotel_visualization():
    """Test visualization using Hotel dataset"""
    
    # 1. Load Hotel dataset
    hotel_data_path = os.path.join(ROOT_DIR, 'datasets/hotel/test')
    obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = load_hotel_trajectories(hotel_data_path)
    
    print(f"\nTrajectory shapes:")
    print(f"obs_traj: {obs_traj.shape}")
    print(f"pred_traj: {pred_traj.shape}")
    
    # 2. Initialize GroupAssignment with smaller threshold for tighter groups
    group_assignment = GroupAssignment(d_type='euclidean', th=1.0)  # 1 meter threshold
    
    # 3. Get group assignments
    v_rel = obs_traj_rel.permute(0, 3, 1, 2)  # [B, 2, T, N]
    v_abs = obs_traj.permute(0, 3, 1, 2)      # [B, 2, T, N]
    
    v_grouped, group_indices, dist_mat, result_dict = group_assignment(v_rel, v_abs)
    print(f"\nGroup assignments shape: {group_indices.shape}")
    print(f"Group assignments: {group_indices}")
    print(f"\nNumber of pedestrians: {obs_traj.size(2)}")
    print(f"Number of unique groups: {len(torch.unique(group_indices))}")
    
    # 4. Create predictions with uncertainty
    B, T, N, _ = pred_traj.shape
    V_pred_viz = torch.zeros(B, T, N, 5)  # [B, T, N, 5]
    
    # Copy ground truth as mean prediction
    V_pred_viz[..., :2] = pred_traj
    
    # Add uncertainty parameters
    V_pred_viz[..., 2] = -2.0  # log(variance) ≈ log(0.135)
    V_pred_viz[..., 3] = 0.0   # Zero correlation (pre-tanh)
    V_pred_viz[..., 4] = 0.0   # Unit axis ratio (circular uncertainty)
    
    # 5. Create visualization
    # Static visualization
    fig_img = data_visualizer(
        V_pred_viz,  # [B, T, N, 5] prediction with uncertainty
        obs_traj,    # [B, T_obs, N, 2] observed trajectory
        pred_traj,   # [B, T_pred, N, 2] ground truth predictions
        group_indices=group_indices  # [B, N] group assignments
    )
    
    # Save static visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.imshow(fig_img[:, :, :3])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('hotel_visualization.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print("\nStatic visualization saved as 'hotel_visualization.png'")
    
    # Create video visualization
    # Use original dataset framerate (2.5 Hz)
    create_trajectory_video(
        V_pred_viz,
        obs_traj,
        pred_traj,
        group_indices=group_indices,
        output_path='hotel_visualization.mp4',
        fps=2.5  # Original dataset sampling rate
    )
    
    print("Video visualization saved as 'hotel_visualization.mp4'")
    return "Test completed successfully!"

if __name__ == "__main__":
    print(test_hotel_visualization())