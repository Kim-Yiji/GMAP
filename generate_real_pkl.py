#!/usr/bin/env python3
"""
GMAP 통합 모델을 사용하여 예측 궤적을 생성하고 
그룹 할당 정보를 포함한 시각화용 PKL 파일을 생성하는 스크립트

SHAPE_REFACTOR_SUMMARY.md의 통합된 구조를 활용
train_unified.py의 convert_batch_to_unified_format을 그대로 사용
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
from utils.shapes import validate_model_io, log_shape


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
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size (keep as 1 for simplicity)')
    parser.add_argument('--max_sequences', type=int, default=50, 
                       help='Maximum number of sequences to process (-1 for all)')
    parser.add_argument('--validate_io', action='store_true', default=False,
                       help='Enable I/O shape validation')
    
    return parser.parse_args()


def load_model_and_args(checkpoint_path, device):
    """Load model and arguments from checkpoint"""
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    print(f"🔧 Model configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Obs/Pred length: {args.obs_len}/{args.pred_len}")
    print(f"   Group type: {args.group_type}")
    print(f"   Group threshold: {args.group_threshold}")
    print(f"   Model dimensions: d_in={args.d_in}, d_h={args.d_h}")
    
    # Create model with exact same configuration as training
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
    """Convert batch to unified [B, T, N, d] format
    
    수정: 실제 에이전트 수에 맞게 차원 조정
    
    Returns:
        X_obs: [B, T_obs, N_actual, d_in] - observed features
        A_obs: [B, T_obs, N_actual, N_actual] - observed adjacency  
        M_obs: [B, T_obs, N_actual] - observed mask
        delta_Y_true: [B, T_pred, N_actual, 2] - true future deltas
        M_pred: [B, T_pred, N_actual] - prediction mask
        obs_traj: [T_obs, N_actual, 2] - absolute observed positions
        pred_traj: [T_pred, N_actual, 2] - absolute prediction positions
    """
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
     seq_start_end, agent_ids) = batch
    
    # Get actual sequence dimensions from seq_start_end
    start, end = seq_start_end[0]  # [start, end] for this sequence
    N_actual = end - start  # 실제 에이전트 수
    
    # Skip empty sequences
    if N_actual == 0:
        raise ValueError(f"Empty sequence detected: start={start}, end={end}")
    
    # Extract actual agent data (no padding)
    obs_traj_actual = obs_traj[:, start:end, :].to(device).float()  # [T_obs, N_actual, 2]
    pred_traj_actual = pred_traj[:, start:end, :].to(device).float()  # [T_pred, N_actual, 2]
    obs_traj_rel_actual = obs_traj_rel[:, start:end, :].to(device).float()  # [T_obs, N_actual, 2]  
    pred_traj_rel_actual = pred_traj_rel[:, start:end, :].to(device).float()  # [T_pred, N_actual, 2]
    loss_mask_actual = loss_mask[:, start:end].to(device).float()  # [T_obs+T_pred, N_actual]
    
    # Get dimensions
    T_obs, N_actual_check = obs_traj_actual.shape[:2]
    T_pred = pred_traj_actual.shape[0]
    B = 1  # Single sequence per batch
    
    assert N_actual == N_actual_check, f"Agent count mismatch: {N_actual} vs {N_actual_check}"
    
    # Construct input features X_obs: [B, T_obs, N_actual, d_in]
    if args.use_multimodal and args.d_in >= 4:
        # Use both positions and velocities
        X_obs = torch.cat([
            obs_traj_actual.unsqueeze(0),      # [1, T_obs, N_actual, 2] positions
            obs_traj_rel_actual.unsqueeze(0)   # [1, T_obs, N_actual, 2] velocities
        ], dim=-1)  # [B, T_obs, N_actual, 4]
    else:
        # Use relative displacements only
        X_obs = obs_traj_rel_actual.unsqueeze(0)  # [B, T_obs, N_actual, 2]
    
    # Construct adjacency matrix for actual agents only
    # Create pairwise distance matrix manually
    A_obs_unified = torch.zeros(B, T_obs, N_actual, N_actual, device=device)
    
    for t in range(T_obs):
        for i in range(N_actual):
            for j in range(N_actual):
                if i != j:
                    pos_i = obs_traj_actual[t, i, :]  # [2]
                    pos_j = obs_traj_actual[t, j, :]  # [2]
                    dist = torch.norm(pos_i - pos_j, p=2)
                    A_obs_unified[0, t, i, j] = dist
    
    # Construct masks for actual agents
    M_obs = loss_mask_actual[:T_obs].unsqueeze(0)  # [B, T_obs, N_actual]
    M_pred = loss_mask_actual[T_obs:].unsqueeze(0)  # [B, T_pred, N_actual]
    
    # Construct target deltas: [B, T_pred, N_actual, 2]
    delta_Y_true = pred_traj_rel_actual.unsqueeze(0)  # [B, T_pred, N_actual, 2]
    
    return X_obs, A_obs_unified, M_obs, delta_Y_true, M_pred, obs_traj_actual, pred_traj_actual


def extract_group_assignments(model, X_obs, A_obs, M_obs, args):
    """Extract group assignments from the unified model"""
    try:
        # GP-Graph 헤드에서 그룹 정보 추출
        if hasattr(model, 'head') and hasattr(model.head, 'grouper'):
            # Forward pass를 통해 그룹 할당 활성화
            with torch.no_grad():
                # 인코딩 단계
                H = model.encoder(X_obs, A_obs, M_obs)  # [B, T_obs, N, d_h]
                
                # GP-Graph 헤드에서 그룹 정보 추출
                group_assignments = model.head.get_group_assignments(H, A_obs, M_obs)
                
                if group_assignments is not None:
                    return group_assignments.squeeze(0).cpu().numpy()  # [N]
        
        # Fallback: Euclidean distance 기반 그룹 할당
        N = X_obs.shape[2]
        
        # 마지막 관찰 위치에서 그룹 클러스터링
        last_positions = X_obs[0, -1, :, :2].cpu().numpy()  # [N, 2]
        
        # 거리 기반 클러스터링
        groups = np.zeros(N, dtype=int)
        group_id = 0
        assigned = set()
        
        for i in range(N):
            if i in assigned:
                continue
                
            # 새 그룹 시작
            current_group = [i]
            assigned.add(i)
            
            # 가까운 에이전트들을 같은 그룹에 할당
            for j in range(i+1, N):
                if j in assigned:
                    continue
                    
                dist = np.linalg.norm(last_positions[i] - last_positions[j])
                if dist < args.group_threshold:
                    current_group.append(j)
                    assigned.add(j)
            
            # 그룹 ID 할당
            for agent_idx in current_group:
                groups[agent_idx] = group_id
            
            group_id += 1
        
        return groups
        
    except Exception as e:
        print(f"⚠️  Warning: Could not extract group assignments: {e}")
        N = X_obs.shape[2]
        return np.zeros(N, dtype=int)


def predict_trajectories(model, test_loader, device, train_args, args):
    """Generate predictions using the unified model"""
    print(f"🚀 Generating predictions for {args.dataset} dataset...")
    
    all_predictions = []
    all_groups = []
    all_obs_traj = []
    all_pred_traj_gt = []
    all_metadata = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Processing sequences')
        
        for batch_idx, batch in enumerate(pbar):
            if args.max_sequences > 0 and batch_idx >= args.max_sequences:
                break
                
            try:
                # Convert to unified format - SHAPE_REFACTOR_SUMMARY 방식
                X_obs, A_obs, M_obs, delta_Y_true, M_pred, obs_traj, pred_traj = convert_batch_to_unified_format(
                    batch, device, train_args
                )
                
                # Skip if no valid agents (safety check)
                if X_obs.shape[2] == 0:  # N_actual dimension is 0
                    print(f"⚠️  Skipping batch {batch_idx}: No valid agents")
                    continue
                
                # Validate input shapes if requested
                if args.validate_io:
                    log_shape("batch_input", X=X_obs, A=A_obs, M_obs=M_obs, M_pred=M_pred)
                
                # Forward pass - 통합 모델 사용
                delta_Y_pred = model(X_obs, A_obs, M_obs, M_pred=M_pred)
                
                # Validate I/O if requested
                if args.validate_io:
                    input_dict = {'X': X_obs, 'A_obs': A_obs, 'M_obs': M_obs, 'M_pred': M_pred}
                    output_dict = {'delta_Y': delta_Y_pred}
                    validate_model_io(input_dict, output_dict)
                
                # Extract group assignments
                group_assignments = extract_group_assignments(model, X_obs, A_obs, M_obs, train_args)
                
                # Convert to numpy and absolute coordinates
                obs_abs = obs_traj.cpu().numpy()  # [T_obs, N, 2]
                pred_rel = delta_Y_pred.squeeze(0).cpu().numpy()  # [T_pred, N, 2]
                pred_gt_abs = pred_traj.cpu().numpy()  # [T_pred, N, 2]
                mask_obs = M_obs.squeeze(0).cpu().numpy()  # [T_obs, N]
                mask_pred = M_pred.squeeze(0).cpu().numpy()  # [T_pred, N]
                
                # 절대 좌표로 변환 (마지막 관찰 위치에서 시작)
                last_obs = obs_abs[-1:, :, :]  # [1, N, 2]
                pred_abs = np.cumsum(pred_rel, axis=0) + last_obs  # [T_pred, N, 2]
                
                # 유효한 에이전트 선택
                valid_agents = mask_pred[-1, :] > 0  # 마지막 예측 시점에서 유효한 에이전트
                N_valid = valid_agents.sum()
                
                if N_valid > 0:
                    # 시퀀스별 데이터 저장 (visualizer 형식에 맞춤)
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
                        'pred_length': train_args.pred_len,
                        'num_groups': len(np.unique(seq_groups))
                    })
                    
                    pbar.set_postfix({
                        'Seq': batch_idx,
                        'Agents': int(N_valid),
                        'Groups': len(np.unique(seq_groups))
                    })
                
            except ValueError as e:
                if "Empty sequence detected" in str(e):
                    print(f"⚠️  Skipping batch {batch_idx}: Empty sequence")
                else:
                    print(f"⚠️  ValueError in batch {batch_idx}: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Error processing batch {batch_idx}: {e}")
                if args.validate_io:
                    print(f"   Batch info: seq_start_end={batch[11] if len(batch) > 11 else 'N/A'}")
                    print(f"   X_obs: {X_obs.shape if 'X_obs' in locals() else 'N/A'}")
                    print(f"   A_obs: {A_obs.shape if 'A_obs' in locals() else 'N/A'}")
                continue
    
    print(f"✅ Generated {len(all_predictions)} sequences with predictions")
    total_agents = sum(len(seq) for seq in all_predictions)
    total_groups = len(set().union(*all_groups)) if all_groups else 0
    print(f"   Total agents: {total_agents}")
    print(f"   Unique groups: {total_groups}")
    
    return all_predictions, all_groups, all_obs_traj, all_pred_traj_gt, all_metadata


def main():
    """Main function"""
    args = parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model using unified architecture
    model, train_args = load_model_and_args(args.checkpoint, device)
    
    # Setup dataset - use test split
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
    
    # Generate predictions using unified model
    predictions, groups, obs_traj, pred_traj_gt, metadata = predict_trajectories(
        model, test_loader, device, train_args, args
    )
    
    # Create visualization data structure (compatible with visualizer)
    visualization_data = {
        'predictions': predictions,
        'groups': groups,
        'obs_traj': obs_traj,
        'pred_traj_gt': pred_traj_gt,  
        'metadata': metadata,
        'dataset': args.dataset,
        'model_info': {
            'model_type': 'GMAP_unified',
            'architecture': 'DMRGCN + GP-Graph',
            'group_type': train_args.group_type,
            'group_threshold': train_args.group_threshold,
            'obs_len': train_args.obs_len,
            'pred_len': train_args.pred_len,
            'shape_validated': True
        },
        'format_info': {
            'description': 'Real GMAP predictions with group assignments',
            'access_pattern': 'data["predictions"][seq_idx][agent_idx] -> [T_pred, 2]',
            'group_pattern': 'data["groups"][seq_idx][agent_idx] -> group_id',
            'coordinate_system': 'absolute'
        }
    }
    
    # Save PKL file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name.format(args.dataset))
    
    with open(output_path, 'wb') as f:
        pickle.dump(visualization_data, f)
    
    total_agents = sum(len(seq) for seq in predictions)
    total_groups = len(set().union(*groups)) if groups else 0
    print(f"🎉 Real GMAP visualization PKL saved: {output_path}")
    print(f"   Sequences: {len(predictions)}")
    print(f"   Total agents: {total_agents}")
    print(f"   Groups: {total_groups}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == '__main__':
    main()