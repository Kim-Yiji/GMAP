#!/usr/bin/env python3
"""
GMAP 통합 모델을 사용하여 예측 궤적을 생성하고 
그룹 할당 정보를 포함한 시각화용 PKL 파일을 생성하는 스크립트 (수정 버전)

train_unified.py의 구조를 따라서 작성함
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
    parser.add_argument('--checkpoint', default='./server_exp-hotel/hotel_best.pth',
                       help='Path to model checkpoint')
    
    # Output parameters
    parser.add_argument('--output_dir', default='../ETH-UCY-Trajectory-Visualizer/pred_traj_dump/',
                       help='Output directory for PKL files')
    parser.add_argument('--output_name', default='GMAP_{}.pkl',
                       help='Output PKL filename pattern')
    
    # Prediction parameters
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size (keep as 1 for simplicity)')
    parser.add_argument('--max_sequences', type=int, default=50, 
                       help='Maximum number of sequences to process (-1 for all)')
    
    return parser.parse_args()


def load_model_and_args(checkpoint_path, device):
    """Load model and arguments from checkpoint - train_unified.py 스타일"""
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    print(f"🔧 Model configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Obs/Pred length: {args.obs_len}/{args.pred_len}")
    print(f"   Group type: {args.group_type}")
    print(f"   Group threshold: {args.group_threshold}")
    
    # Create model with same configuration as training
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
        enable_paths={
            'agent': args.enable_agent,
            'intra': args.enable_intra,
            'inter': args.enable_inter
        },
        use_multimodal=args.use_multimodal,
        use_simple_head=args.use_simple_head
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model, args


def convert_batch_to_unified_format(batch, device, args):
    """Convert batch to unified format - train_unified.py와 동일"""
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
    
    # Get dimensions
    T_obs, N = obs_traj.shape[:2]
    T_pred = pred_traj.shape[0]
    B = 1  # Single sequence per batch
    
    # Construct input features X_obs: [B, T_obs, N, d_in]
    if args.use_multimodal and args.d_in >= 4:
        # Use both positions and velocities
        X_obs = torch.cat([
            obs_traj.unsqueeze(0),      # [1, T_obs, N, 2] positions
            obs_traj_rel.unsqueeze(0)   # [1, T_obs, N, 2] velocities
        ], dim=-1)  # [B, T_obs, N, 4]
    else:
        # Use relative displacements only
        X_obs = obs_traj_rel.unsqueeze(0)  # [B, T_obs, N, 2]
    
    # Construct adjacency A_obs: [B, T_obs, N, N]
    A_obs_unified = A_obs[:, 1, :, :, :].permute(0, 1, 2, 3)  # [B, T_obs, N, N]
    
    # Construct masks
    M_obs = loss_mask[:T_obs].unsqueeze(0)  # [B, T_obs, N]
    M_pred = loss_mask[T_obs:].unsqueeze(0)  # [B, T_pred, N]
    
    # Return data for prediction and visualization
    return X_obs, A_obs_unified, M_obs, M_pred, obs_traj, pred_traj_rel, loss_mask


def extract_group_assignments(model, X_obs, A_obs, M_obs, args):
    """Extract group assignments from the model"""
    try:
        # GP-Graph 모델에서 그룹 정보 추출 시도
        if hasattr(model, 'gpgraph_head'):
            # GP-Graph 헤드에서 그룹 정보를 얻기 위해 forward pass 실행
            with torch.no_grad():
                _ = model(X_obs, A_obs, M_obs)
                
                # 그룹 할당 정보 추출
                if hasattr(model.gpgraph_head, 'group_assignments'):
                    group_assignments = model.gpgraph_head.group_assignments
                    return group_assignments.cpu().numpy()
        
        # 기본 그룹 할당: Euclidean distance 기반
        N = X_obs.shape[2]
        last_positions = X_obs[0, -1, :, :2].cpu().numpy()  # [N, 2] - 마지막 관찰 위치
        
        # 간단한 클러스터링 (거리 기반)
        groups = []
        group_id = 0
        assigned = set()
        
        for i in range(N):
            if i in assigned:
                continue
                
            current_group = [i]
            assigned.add(i)
            
            for j in range(i+1, N):
                if j in assigned:
                    continue
                    
                dist = np.linalg.norm(last_positions[i] - last_positions[j])
                if dist < args.group_threshold:
                    current_group.append(j)
                    assigned.add(j)
            
            # 그룹 할당
            for agent_idx in current_group:
                groups.append((agent_idx, group_id))
            
            group_id += 1
        
        # 정렬해서 agent_id별 그룹 반환
        groups.sort(key=lambda x: x[0])
        group_assignments = np.array([g[1] for g in groups])
        
        return group_assignments
        
    except Exception as e:
        print(f"⚠️  Warning: Could not extract group assignments: {e}")
        N = X_obs.shape[2]
        return np.zeros(N, dtype=int)  # 모든 에이전트를 그룹 0으로 설정


def predict_trajectories(model, test_loader, device, train_args, max_sequences):
    """Generate predictions for test sequences"""
    print(f"🚀 Generating predictions for {train_args.dataset} dataset...")
    
    all_predictions = []
    all_groups = []
    all_obs_traj = []
    all_pred_traj_gt = []
    all_metadata = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Predicting trajectories')
        
        for batch_idx, batch in enumerate(pbar):
            if max_sequences > 0 and batch_idx >= max_sequences:
                break
                
            try:
                # Convert to unified format
                X_obs, A_obs, M_obs, M_pred, obs_traj, pred_traj_rel, loss_mask = convert_batch_to_unified_format(
                    batch, device, train_args
                )
                
                # Forward pass to get predictions
                delta_Y_pred = model(X_obs, A_obs, M_obs, M_pred=M_pred)
                
                # Extract group assignments
                group_assignments = extract_group_assignments(model, X_obs, A_obs, M_obs, train_args)
                
                # Convert predictions to absolute coordinates
                obs_abs = obs_traj.cpu().numpy()  # [T_obs, N, 2]
                pred_rel = delta_Y_pred.squeeze(0).cpu().numpy()  # [T_pred, N, 2]
                pred_gt_rel = pred_traj_rel.cpu().numpy()  # [T_pred, N, 2]
                mask = loss_mask.cpu().numpy()  # [T_obs + T_pred, N]
                
                # 마지막 관찰 위치에서 시작해서 누적 합으로 절대 좌표 계산
                last_obs = obs_abs[-1:, :, :]  # [1, N, 2]
                pred_abs = np.cumsum(pred_rel, axis=0) + last_obs  # [T_pred, N, 2]
                pred_gt_abs = np.cumsum(pred_gt_rel, axis=0) + last_obs  # [T_pred, N, 2]
                
                # 유효한 에이전트만 선택 (mask 기반)
                valid_agents = mask[-1, :] > 0  # 마지막 시점에서 유효한 에이전트
                N_valid = valid_agents.sum()
                
                if N_valid > 0:
                    # 시퀀스별 데이터 저장 (각 에이전트별로)
                    seq_predictions = []
                    seq_groups = []
                    
                    for agent_idx in range(len(valid_agents)):
                        if valid_agents[agent_idx]:
                            seq_predictions.append(pred_abs[:, agent_idx, :])  # [T_pred, 2]
                            seq_groups.append(group_assignments[agent_idx])
                    
                    all_predictions.append(seq_predictions)
                    all_groups.append(np.array(seq_groups))
                    all_obs_traj.append(obs_abs[:, valid_agents, :])  # [T_obs, N_valid, 2]
                    all_pred_traj_gt.append(pred_gt_abs[:, valid_agents, :])  # [T_pred, N_valid, 2]
                    
                    all_metadata.append({
                        'seq_idx': batch_idx,
                        'num_agents': int(N_valid),
                        'pred_length': train_args.pred_len
                    })
                    
                    pbar.set_postfix({
                        'Seq': batch_idx,
                        'Agents': int(N_valid),
                        'Groups': len(np.unique(seq_groups))
                    })
                
            except Exception as e:
                print(f"⚠️  Error processing batch {batch_idx}: {e}")
                continue
    
    print(f"✅ Generated {len(all_predictions)} sequences with predictions")
    return all_predictions, all_groups, all_obs_traj, all_pred_traj_gt, all_metadata


def main():
    """Main function"""
    args = parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    model, train_args = load_model_and_args(args.checkpoint, device)
    
    # Setup dataset - use test directory
    dataset_path = f'./datasets/{args.dataset}/test/'
    print(f"📁 Loading dataset from {dataset_path}")
    
    test_dataset = TrajectoryDataset(
        dataset_path,
        obs_len=train_args.obs_len,
        pred_len=train_args.pred_len,
        skip=train_args.skip,
        min_ped=train_args.min_ped,
        use_cache=train_args.use_cache,
        cache_dir=train_args.cache_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"📊 Test dataset: {len(test_dataset)} sequences")
    
    # Generate predictions
    predictions, groups, obs_traj, pred_traj_gt, metadata = predict_trajectories(
        model, test_loader, device, train_args, args.max_sequences
    )
    
    # Create visualization data structure
    visualization_data = {
        'predictions': predictions,
        'groups': groups,
        'obs_traj': obs_traj,
        'pred_traj_gt': pred_traj_gt,
        'metadata': metadata,
        'dataset': args.dataset,
        'model_info': {
            'model_type': 'GMAP_unified',
            'group_type': train_args.group_type,
            'group_threshold': train_args.group_threshold,
            'obs_len': train_args.obs_len,
            'pred_len': train_args.pred_len
        }
    }
    
    # Save PKL file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name.format(args.dataset))
    
    with open(output_path, 'wb') as f:
        pickle.dump(visualization_data, f)
    
    total_agents = sum(len(seq) for seq in predictions)
    print(f"🎉 Visualization PKL saved: {output_path}")
    print(f"   Sequences: {len(predictions)}")
    print(f"   Total agents: {total_agents}")
    print(f"   Groups: {len(set().union(*groups)) if groups else 0}")


if __name__ == '__main__':
    main()