import os
import sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
import cv2

# Add the root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from datasets.dataloader import TrajectoryDataset, collate_fn
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
from utils.visualization import create_static_visualization, create_trajectory_video
import matplotlib.pyplot as plt
import cv2

def load_model_and_args(checkpoint_path):
    """Load trained model and its arguments"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    
    # Setup model
    model = DMRGCN_GPGraph_Model(
        d_in=args.input_dim,
        d_h=args.hidden_dims[-1],
        d_gp_in=args.hidden_dims[-1],
        T_pred=args.pred_len,
        output_dim=2,
        dmrgcn_hidden_dims=args.hidden_dims,
        dmrgcn_kernel_size=tuple(args.kernel_size),
        dmrgcn_dropout=args.dropout,
        group_type=args.group_type,
        group_threshold=args.group_th,
        mix_type=args.mix_type,
        enable_paths={'agent': args.enable_agent,
                     'intra': args.enable_intra,
                     'inter': args.enable_inter},
        distance_scales=args.distance_scales,
        share_backbone=args.share_backbone
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, args

def visualize_predictions_with_groups(model, dataset_path, save_dir='./visualization_outputs',
                                    num_sequences=5):
    """Visualize trajectory predictions with learned grouping"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    dataset = TrajectoryDataset(
        dataset_path,
        obs_len=model.obs_len,
        pred_len=model.pred_len,
        skip=1,
        delim='\t'
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Process sequences
    for seq_idx, batch in enumerate(loader):
        if seq_idx >= num_sequences:
            break
            
        # Unpack batch
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, non_linear_ped, loss_mask, _, _, _, _, seq_start_end = batch
        
        # Create observation mask
        obs_mask = torch.ones(obs_traj.shape[1], obs_traj.shape[0], device=obs_traj.device)  # [T, N]
        obs_mask = obs_mask.transpose(0, 1).unsqueeze(0)  # [1, N, T]
        
        # Create adjacency matrix (fully connected for simplicity)
        N = obs_traj.shape[1]
        adj_mat = torch.ones(1, N, N, device=obs_traj.device)
        
        # Prepare input
        V_obs = obs_traj.permute(1, 0, 2).unsqueeze(0)  # [1, T, N, 2]
        
        # Forward pass through model with grouping
        with torch.no_grad():
            pred_traj_fake, group_indices = model(
                V_obs, adj_mat, obs_mask,
                return_groups=True
            )
        
        # Create static visualization
        fig_img = create_static_visualization(
            obs_traj.permute(1, 0, 2).unsqueeze(0),  # [1, T, N, 2]
            pred_traj_fake,
            group_indices,
            title=f"Sequence {seq_idx + 1}"
        )
        
        # Save static visualization
        cv2.imwrite(
            os.path.join(save_dir, f'sequence_{seq_idx}_static.png'),
            cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR)
        )
        
        # Create video visualization
        create_trajectory_video(
            obs_traj.permute(1, 0, 2).unsqueeze(0),  # [1, T, N, 2]
            pred_traj_fake,
            group_indices,
            output_path=os.path.join(save_dir, f'sequence_{seq_idx}_dynamic.mp4'),
            fps=2.5  # Original dataset framerate
        )
        
        # Save grouping information
        group_info = {
            'group_indices': group_indices.cpu().numpy(),
            'sequence_id': seq_idx,
            'seq_start_end': seq_start_end.cpu().numpy()
        }
        with open(os.path.join(save_dir, f'sequence_{seq_idx}_groups.pkl'), 'wb') as f:
            pickle.dump(group_info, f)
        
        print(f"Processed sequence {seq_idx + 1}/{num_sequences}")

def main():
    # Default model configuration
    class DefaultArgs:
        def __init__(self):
            # Model dimensions
            self.input_dim = 2
            self.hidden_dims = [64, 64, 64, 64, 128]
            self.kernel_size = [3, 1]
            
            # Training parameters
            self.dropout = 0.1
            self.distance_scales = [0.5, 1.0, 2.0]
            
            # Grouping parameters
            self.group_type = 'euclidean'
            self.group_th = 2.0
            self.mix_type = 'mean'
            
            # Path configuration
            self.enable_agent = True
            self.enable_intra = True
            self.enable_inter = True
            
            # Model behavior
            self.share_backbone = True
            
            # Sequence lengths
            self.obs_len = 8
            self.pred_len = 12
    
    # Load model with default configuration
    args = DefaultArgs()
    checkpoint_path = './server_exp-hotel/checkpoint_epoch_4.pth'
    model = DMRGCN_GPGraph_Model(
        d_in=args.input_dim,
        d_h=args.hidden_dims[-1],
        d_gp_in=args.hidden_dims[-1],
        T_pred=args.pred_len,
        output_dim=2,
        dmrgcn_hidden_dims=args.hidden_dims,
        dmrgcn_kernel_size=tuple(args.kernel_size),
        dmrgcn_dropout=args.dropout,
        group_type=args.group_type,
        group_threshold=args.group_th,
        mix_type=args.mix_type,
        enable_paths={'agent': args.enable_agent,
                     'intra': args.enable_intra,
                     'inter': args.enable_inter},
        distance_scales=args.distance_scales,
        share_backbone=args.share_backbone
    )
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Visualize test sequences
    dataset_path = os.path.join(ROOT_DIR, 'datasets/hotel/test')
    visualize_predictions_with_groups(
        model,
        dataset_path,
        save_dir='./model_visualization_outputs',
        num_sequences=5
    )

if __name__ == "__main__":
    main()