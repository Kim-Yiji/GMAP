import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from model.gpgraph_adapter import GroupAssignment
from utils.group_data import save_group_data, load_group_data, numpy_to_torch
from copy_dmrgcn.utils.visualizer import data_visualizer

def test_group_visualization():
    # 1. Create test data
    batch_size = 1
    seq_len = 8
    obs_len = 4
    pred_len = 4
    num_peds = 5
    
    # Generate trajectories [B, T, N, 2]
    v_abs = torch.zeros(batch_size, seq_len, num_peds, 2)
    t = torch.linspace(0, 2*3.14159, seq_len).unsqueeze(-1)
    # Create two groups of pedestrians with different centers
    centers = torch.tensor([[0, 0], [0, 2], [0, 2], [-2, -1], [-2, -1]])
    radii = torch.tensor([1.0, 1.2, 0.8, 1.5, 0.7])
    
    for i in range(num_peds):
        v_abs[0, :, i, 0] = centers[i, 0] + radii[i] * torch.cos(t).squeeze()
        v_abs[0, :, i, 1] = centers[i, 1] + radii[i] * torch.sin(t).squeeze()
    
    # Add some noise
    v_abs = v_abs + torch.randn_like(v_abs) * 0.1
    
    # Create relative displacements [B, T, N, 2]
    v_rel = torch.zeros_like(v_abs)
    v_rel[:, 1:] = v_abs[:, 1:] - v_abs[:, :-1]
    
    # Split into observation and prediction trajectories
    obs_traj = v_abs[:, :obs_len]  # [B, T_obs, N, 2]
    pred_traj_gt = v_abs[:, obs_len:]  # [B, T_pred, N, 2]
    
    # Create mock predictions with uncertainty (shape: B, T, N, 5)
    # For each pedestrian at each timestep:
    # - First 2 values: predicted mean position
    # - Value 3: log variance
    # - Value 4: correlation (will be passed through tanh)
    # - Value 5: log axis ratio
    V_pred = torch.zeros(batch_size, pred_len, num_peds, 5)
    V_pred[..., :2] = pred_traj_gt  # Copy ground truth as mean prediction
    V_pred[..., 2] = -2.0  # log(variance) ≈ log(0.135)
    V_pred[..., 3] = 0.0   # Zero correlation (pre-tanh)
    V_pred[..., 4] = 0.0   # Unit axis ratio (circular uncertainty)os
import matplotlib.pyplot as plt

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpgraph_adapter import GroupAssignment
from utils.group_data import save_group_data, load_group_data, numpy_to_torch
from copy_dmrgcn.utils.visualizer import visualize_scene, data_visualizer
from torch.utils.data import DataLoader

def test_group_visualization():
    # 1. Create test data
    batch_size = 1
    seq_len = 8
    obs_len = 4
    pred_len = 4
    num_peds = 5
    
    # Sample test data to simulate trajectories
    v_rel = torch.randn(batch_size, seq_len, num_peds, 2)  # [B, T, N, 2]
    v_abs = torch.randn(batch_size, seq_len, num_peds, 2)  # [B, T, N, 2]
    
    # Create observation and prediction trajectories
    obs_traj = v_abs[:, :obs_len]  # [B, T_obs, N, 2]
    pred_traj_gt = v_abs[:, obs_len:]  # [B, T_pred, N, 2]
    
    # Create mock prediction with uncertainty (shape: B, T, N, 5)
    V_pred = torch.randn(batch_size, pred_len, num_peds, 5)
    
    # Ensure v_rel and v_abs have the right format for GroupAssignment
    # GroupAssignment expects [B, 2, T, N]
    v_rel_ga = v_rel.permute(0, 3, 1, 2)  # [B, T, N, 2] -> [B, 2, T, N]
    v_abs_ga = v_abs.permute(0, 3, 1, 2)  # [B, T, N, 2] -> [B, 2, T, N]
    
    # 2. Initialize GroupAssignment with smaller threshold for more groups
    group_assignment = GroupAssignment(d_type='euclidean', th=0.5)  # Smaller threshold = more groups
    
    # 3. Get group assignments
    v_grouped, group_indices, dist_mat, result_dict = group_assignment(v_rel_ga, v_abs_ga)
    print(f"\nInput shapes:")
    print(f"v_rel shape: {v_rel.shape}")
    print(f"v_abs shape: {v_abs.shape}")
    print(f"group_indices shape: {group_indices.shape}")
    
    # 4. Save results
    filepath = save_group_data(result_dict, dataset_name='test')
    print(f"Saved group data to: {filepath}")
    
    # 5. Load and convert results
    loaded_dict = load_group_data(filepath)
    torch_dict = numpy_to_torch(loaded_dict)
    
    print("\nGroup assignments:", torch_dict['group_indices'])
    print("\nNumber of unique groups:", len(torch.unique(torch_dict['group_indices'])))
    
    from copy_dmrgcn.utils.video_visualizer import create_trajectory_video
    
    # Create static visualization
    fig_img = data_visualizer(
        V_pred,  # [B, T, N, 5] prediction with uncertainty
        obs_traj,  # [B, T_obs, N, 2] observed trajectory
        pred_traj_gt,  # [B, T_pred, N, 2] ground truth predictions
        group_indices=group_indices  # [B, N] group assignments
    )
    
    # Save static visualization
    plt.figure(figsize=(10, 7))
    plt.imshow(fig_img[:, :, :3])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('test_visualization.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print("\nStatic visualization saved as 'test_visualization.png'")
    
    # Create video visualization
    create_trajectory_video(
        V_pred,
        obs_traj,
        pred_traj_gt,
        group_indices=group_indices,
        output_path='test_visualization.mp4',
        fps=10
    )
    
    print("Video visualization saved as 'test_visualization.mp4'")
    return "Test completed successfully!"

if __name__ == "__main__":
    print(test_group_visualization())