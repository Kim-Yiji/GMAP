#!/usr/bin/env python3
"""
GMAP 모델 데이터 로딩 및 형태 확인 스크립트
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

# Import unified model and utilities
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
from datasets.dataloader import TrajectoryDataset, collate_fn


def test_data_loading():
    """데이터 로딩 및 형태 확인"""
    print("🔍 데이터 로딩 테스트 시작")
    
    # Setup data loader
    dataset_path = './copy_dmrgcn/datasets/hotel/'
    test_dataset = TrajectoryDataset(
        dataset_path + 'test/',
        obs_len=8,
        pred_len=12,
        skip=1,
        min_ped=1,
        delim='tab',
        use_cache=True,
        cache_dir='./data_cache/hotel'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✅ 데이터셋 로딩 완료: {len(test_loader)} 시퀀스")
    
    # 첫 번째 배치 확인
    for batch_idx, batch in enumerate(test_loader):
        print(f"\n📊 배치 {batch_idx} 형태 확인:")
        
        (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
         non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
         seq_start_end, agent_ids) = batch
        
        print(f"  obs_traj: {obs_traj.shape}")
        print(f"  pred_traj: {pred_traj.shape}")
        print(f"  obs_traj_rel: {obs_traj_rel.shape}")
        print(f"  pred_traj_rel: {pred_traj_rel.shape}")
        print(f"  loss_mask: {loss_mask.shape}")
        print(f"  V_obs: {V_obs.shape}")
        print(f"  A_obs: {A_obs.shape}")
        print(f"  V_pred: {V_pred.shape}")
        print(f"  A_pred: {A_pred.shape}")
        print(f"  seq_start_end: {seq_start_end.shape}")
        print(f"  agent_ids: {agent_ids.shape}")
        
        # 상세 정보
        T_obs, N = obs_traj.shape[:2]
        T_pred = pred_traj.shape[0]
        print(f"\n📏 차원 정보:")
        print(f"  T_obs: {T_obs}, N: {N}, T_pred: {T_pred}")
        
        if batch_idx >= 2:  # 처음 3개만 확인
            break
    
    return test_loader


def test_model_input_conversion(test_loader):
    """모델 입력 변환 테스트"""
    print("\n🔧 모델 입력 변환 테스트")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for batch_idx, batch in enumerate(test_loader):
        print(f"\n📦 배치 {batch_idx} 변환 테스트:")
        
        (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
         non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
         seq_start_end, agent_ids) = batch
        
        # Move to device
        obs_traj = obs_traj.to(device).float()
        obs_traj_rel = obs_traj_rel.to(device).float()
        A_obs = A_obs.to(device).float()
        loss_mask = loss_mask.to(device).float()
        
        # Get dimensions
        T_obs, N = obs_traj.shape[:2]
        T_pred = pred_traj.shape[0]
        B = 1  # Single sequence per batch
        
        print(f"  원본 형태: T_obs={T_obs}, N={N}, T_pred={T_pred}")
        
        # Construct input features X_obs: [B, T_obs, N, d_in]
        X_obs = obs_traj_rel.unsqueeze(0)  # [B, T_obs, N, 2]
        print(f"  X_obs: {X_obs.shape}")
        
        # Construct adjacency A_obs: [B, T_obs, N, N]
        print(f"  A_obs 원본: {A_obs.shape}")
        if len(A_obs.shape) == 5:  # [B, R, T, N, N]
            A_obs_unified = A_obs[:, 1, :, :, :].permute(0, 1, 2, 3)  # [B, T_obs, N, N]
        else:
            A_obs_unified = A_obs
        print(f"  A_obs_unified: {A_obs_unified.shape}")
        
        # Construct masks
        M_obs = loss_mask[:T_obs].unsqueeze(0)  # [B, T_obs, N]
        M_pred = loss_mask[T_obs:].unsqueeze(0)  # [B, T_pred, N]
        print(f"  M_obs: {M_obs.shape}")
        print(f"  M_pred: {M_pred.shape}")
        
        if batch_idx >= 2:  # 처음 3개만 확인
            break


def main():
    """메인 함수"""
    print("🚀 GMAP 데이터 형태 확인 시작")
    
    test_loader = test_data_loading()
    test_model_input_conversion(test_loader)
    
    print("\n✅ 테스트 완료")


if __name__ == '__main__':
    main()