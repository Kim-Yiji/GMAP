#!/usr/bin/env python3
"""
GMAP 모델을 사용하여 예측 궤적을 생성하고 
그룹 할당 정보를 포함한 시각화용 PKL 파일을 생성하는 스크립트

사용법:
python generate_visualization_pkl.py --dataset hotel --checkpoint ./server_exp-hotel/hotel_best.pth
"""

import os
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

# Import unified model and utilities
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
from datasets.dataloader import TrajectoryDataset, collate_fn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Visualization PKL from GMAP Model')
    
    # Model and data parameters
    parser.add_argument('--dataset', default='hotel', 
                       choices=['eth', 'hotel', 'univ', 'zara1', 'zara2'],
                       help='Dataset name')
    parser.add_argument('--checkpoint', required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--obs_len', type=int, default=8, 
                       help='Observation sequence length')
    parser.add_argument('--pred_len', type=int, default=12, 
                       help='Prediction sequence length')
    
    # Output parameters
    parser.add_argument('--output_dir', default='../ETH-UCY-Trajectory-Visualizer/pred_traj_dump/',
                       help='Output directory for PKL files')
    parser.add_argument('--output_name', default='GMAP_{}.pkl',
                       help='Output PKL filename pattern')
    
    # Prediction parameters
    parser.add_argument('--num_samples', type=int, default=1, 
                       help='Number of trajectory samples (use 1 for deterministic)')
    parser.add_argument('--save_groups', action='store_true', default=True,
                       help='Include group assignment information')
    
    return parser.parse_args()


def load_model_and_args(checkpoint_path, device):
    """Load model and arguments from checkpoint"""
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    print(f"🔧 Model configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Obs length: {args.obs_len}")
    print(f"   Pred length: {args.pred_len}")
    print(f"   Group type: {args.group_type}")
    print(f"   Group threshold: {args.group_threshold}")
    
    # Setup model with same configuration as training
    enable_paths = {
        'agent': args.enable_agent,
        'intra': args.enable_intra,
        'inter': args.enable_inter
    }
    
    model = DMRGCN_GPGraph_Model(
        d_in=args.d_in,
        d_h=args.d_h,
        d_gp_in=args.d_gp_in,
        T_pred=args.pred_len,
        output_dim=2,
        dmrgcn_hidden_dims=args.dmrgcn_hidden_dims,
        dmrgcn_kernel_size=tuple(args.kernel_size),
        dmrgcn_dropout=args.dropout,
        distance_scales=args.distance_scales,
        agg_method=args.agg_method,
        group_type=args.group_type,
        group_threshold=args.group_threshold,
        mix_type=args.mix_type,
        enable_paths=enable_paths,
        use_multimodal=args.use_multimodal,
        use_simple_head=args.use_simple_head
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model, args


def convert_batch_to_unified_format(batch, device, args):
    """Convert batch to unified [B, T, N, d] format for GMAP model"""
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
     seq_start_end, agent_ids) = batch
    
    # Move to device
    obs_traj = obs_traj.to(device).float()
    pred_traj = pred_traj.to(device).float()
    obs_traj_rel = obs_traj_rel.to(device).float()
    pred_traj_rel = pred_traj_rel.to(device).float()
    V_obs = V_obs.to(device).float()
    A_obs = A_obs.to(device).float()
    loss_mask = loss_mask.to(device).float()
    
    # Get dimensions - use actual graph dimensions, not padded dimensions
    B, T_obs, N_graph, _ = V_obs.shape  # Use graph dimensions
    T_pred = V_pred.shape[1]
    
    print(f"🔍 디버그: T_obs={T_obs}, N_graph={N_graph}, T_pred={T_pred}")
    print(f"  V_obs: {V_obs.shape}, A_obs: {A_obs.shape}")
    
    # Use the graph-based features and adjacency directly
    # V_obs is already in [B, T, N, d] format
    X_obs = V_obs  # [B, T_obs, N, 2]
    
    # A_obs is in [B, R, T, N, N] format, select distance relation
    A_obs_unified = A_obs[:, 1, :, :, :]  # [B, T_obs, N, N] - distance relation
    
    # Create masks for the actual graph nodes
    M_obs = torch.ones(B, T_obs, N_graph, device=device)  # [B, T_obs, N]
    M_pred = torch.ones(B, T_pred, N_graph, device=device)  # [B, T_pred, N]
    
    # Extract the relevant trajectories for the actual agents
    # We need to map from padded trajectory to graph trajectory
    start, end = seq_start_end[0]  # [start, end] for this sequence
    obs_traj_seq = obs_traj[:, start:end, :]  # [T_obs, N_actual, 2]
    pred_traj_rel_seq = pred_traj_rel[:, start:end, :]  # [T_pred, N_actual, 2]
    
    return X_obs, A_obs_unified, M_obs, M_pred, obs_traj_seq, pred_traj_rel_seq


def predict_trajectories(model, test_loader, device, args):
    """Generate predictions for all test sequences"""
    print(f"🚀 Generating predictions...")
    
    all_predictions = []
    all_groups = []
    all_metadata = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Predicting')
        
        for batch_idx, batch in enumerate(pbar):
            # Convert to unified format
            X_obs, A_obs, M_obs, M_pred, obs_abs, pred_rel_true = convert_batch_to_unified_format(
                batch, device, args
            )
            
            # Forward pass to get predictions and group assignments
            delta_Y_pred = model(X_obs, A_obs, M_obs, M_pred=M_pred)
            
            # Get group assignments if model supports it
            try:
                # Try to get group information from model
                if hasattr(model, 'get_group_assignments'):
                    group_assignments = model.get_group_assignments(X_obs, A_obs, M_obs)
                elif hasattr(model, 'gpgraph_head') and hasattr(model.gpgraph_head, 'group_assignments'):
                    # GP-Graph 모델에서 그룹 정보 추출
                    group_assignments = model.gpgraph_head.group_assignments
                else:
                    # 기본적으로 모든 에이전트를 같은 그룹으로 설정
                    N = X_obs.shape[2]
                    group_assignments = torch.zeros(1, N, dtype=torch.long, device=device)
            except Exception as e:
                print(f"⚠️  Warning: Could not extract group assignments: {e}")
                N = X_obs.shape[2]
                group_assignments = torch.zeros(1, N, dtype=torch.long, device=device)
            
            # Convert predictions to absolute coordinates
            # delta_Y_pred: [B, T_pred, N_graph, 2] -> relative displacements for graph nodes
            # obs_traj_seq: [T_obs, N_actual, 2] -> absolute positions for actual agents
            
            # Get the last observed position for each actual agent
            last_obs_pos = obs_abs[-1, :, :]  # [N_actual, 2] - last observed position
            
            # Convert to numpy and process
            delta_pred_np = delta_Y_pred.squeeze(0).cpu().numpy()  # [T_pred, N_graph, 2]
            last_obs_np = last_obs_pos.cpu().numpy()    # [N_actual, 2]
            group_np = group_assignments.squeeze(0).cpu().numpy()   # [N_graph]
            
            # Make sure we only use predictions for actual agents (N_actual <= N_graph)
            N_actual = last_obs_np.shape[0]
            N_graph = delta_pred_np.shape[1]
            
            print(f"🔍 예측 변환: N_actual={N_actual}, N_graph={N_graph}")
            
            # Take only the predictions for actual agents
            delta_pred_actual = delta_pred_np[:, :N_actual, :]  # [T_pred, N_actual, 2]
            group_actual = group_np[:N_actual]  # [N_actual]
            
            # Convert relative displacements to absolute positions
            pred_abs_np = np.zeros_like(delta_pred_actual)  # [T_pred, N_actual, 2]
            current_pos = last_obs_np.copy()  # [N_actual, 2]
            
            for t in range(delta_pred_actual.shape[0]):
                current_pos += delta_pred_actual[t]  # Add displacement
                pred_abs_np[t] = current_pos.copy()
            
            # Store results for each agent
            seq_predictions = []
            for n in range(pred_abs_np.shape[1]):
                # Extract trajectory for agent n: [T_pred, 2]
                agent_pred = pred_abs_np[:, n, :]  # [T_pred, 2]
                seq_predictions.append(agent_pred)
            
            all_predictions.append(seq_predictions)
            all_groups.append(group_actual)  # Use actual group assignments
            
            # Store metadata
            metadata = {
                'seq_idx': batch_idx,
                'num_agents': pred_abs_np.shape[1],
                'pred_length': pred_abs_np.shape[0]
            }
            all_metadata.append(metadata)
    
    print(f"✅ Generated predictions for {len(all_predictions)} sequences")
    return all_predictions, all_groups, all_metadata


def save_visualization_pkl(predictions, groups, metadata, output_path, dataset_name):
    """Save predictions in format compatible with ETH-UCY-Trajectory-Visualizer"""
    print(f"💾 Saving visualization PKL to {output_path}")
    
    # Create data structure expected by visualizer
    # visualizer expects: data[seq_idx][agent_idx] = trajectory_array
    visualization_data = {
        'predictions': predictions,  # List of sequences, each containing list of agent trajectories
        'groups': groups,           # List of group assignments per sequence
        'metadata': metadata,       # List of metadata per sequence
        'dataset': dataset_name,
        'format_info': {
            'description': 'GMAP trajectory predictions with group information',
            'access_pattern': 'data["predictions"][seq_idx][agent_idx] -> trajectory [T_pred, 2]',
            'group_pattern': 'data["groups"][seq_idx][agent_idx] -> group_id',
            'coordinate_system': 'absolute_world_coordinates'
        }
    }
    
    # Save PKL file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(visualization_data, f)
    
    print(f"✅ Saved {len(predictions)} sequences to {output_path}")
    
    # Print summary
    total_agents = sum(len(seq) for seq in predictions)
    print(f"📊 Summary:")
    print(f"   Total sequences: {len(predictions)}")
    print(f"   Total agents: {total_agents}")
    print(f"   Average agents per sequence: {total_agents/len(predictions):.1f}")
    
    return visualization_data


def main():
    """Main function"""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load model
    model, model_args = load_model_and_args(args.checkpoint, device)
    
    # Setup data loader
    dataset_path = f'./copy_dmrgcn/datasets/{args.dataset}/'
    test_dataset = TrajectoryDataset(
        dataset_path + 'test/',
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=1,
        min_ped=1,
        delim='tab',
        use_cache=model_args.use_cache,
        cache_dir=f'{model_args.cache_dir}/{args.dataset}'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"📊 Dataset info:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Test sequences: {len(test_loader)}")
    print(f"   Observation length: {args.obs_len}")
    print(f"   Prediction length: {args.pred_len}")
    
    # Generate predictions
    predictions, groups, metadata = predict_trajectories(model, test_loader, device, model_args)
    
    # Save results
    output_path = os.path.join(args.output_dir, args.output_name.format(args.dataset))
    visualization_data = save_visualization_pkl(predictions, groups, metadata, output_path, args.dataset)
    
    print(f"🎉 Visualization PKL generation complete!")
    print(f"📁 Output file: {output_path}")
    print(f"🎨 Ready for visualization with ETH-UCY-Trajectory-Visualizer")


if __name__ == '__main__':
    main()