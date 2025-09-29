#!/usr/bin/env python3
"""
GMAP 모델 디버깅 스크립트 - 차원 문제 해결용
"""

import torch
from torch.utils.data import DataLoader
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
from datasets.dataloader import TrajectoryDataset, collate_fn

def debug_model_data_interface():
    """모델과 데이터 간의 차원 불일치 디버깅"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    checkpoint_path = './server_exp-hotel/hotel_best.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    print(f"📋 Model args:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    
    # Load dataset
    dataset_path = './datasets/hotel/test/'
    test_dataset = TrajectoryDataset(
        dataset_path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        min_ped=args.min_ped,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"📊 Dataset info: {len(test_dataset)} sequences")
    
    # Check first batch
    batch = next(iter(test_loader))
    
    print(f"\n🔍 Raw batch data shapes:")
    batch_names = ['obs_traj', 'pred_traj', 'obs_traj_rel', 'pred_traj_rel',
                   'non_linear_ped', 'loss_mask', 'V_obs', 'A_obs', 'V_pred', 'A_pred',
                   'seq_start_end', 'agent_ids']
    
    for i, data in enumerate(batch):
        if i < len(batch_names):
            print(f"   {batch_names[i]}: {data.shape if hasattr(data, 'shape') else type(data)}")
        else:
            print(f"   extra_{i}: {data.shape if hasattr(data, 'shape') else type(data)}")
    
    # Extract the data following train_unified.py format
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
     seq_start_end, agent_ids) = batch
    
    # Move to device
    obs_traj = obs_traj.to(device).float()
    obs_traj_rel = obs_traj_rel.to(device).float()
    V_obs = V_obs.to(device).float()
    A_obs = A_obs.to(device).float()
    loss_mask = loss_mask.to(device).float()
    
    print(f"\\n📐 After device transfer:")
    print(f"   obs_traj: {obs_traj.shape}")
    print(f"   obs_traj_rel: {obs_traj_rel.shape}")
    print(f"   V_obs: {V_obs.shape}")
    print(f"   A_obs: {A_obs.shape}")
    print(f"   loss_mask: {loss_mask.shape}")
    print(f"   seq_start_end: {seq_start_end}")
    
    # Get dimensions
    T_obs, N = obs_traj.shape[:2]
    T_pred = pred_traj.shape[0]
    B = 1  # Single sequence per batch
    
    print(f"\\n📏 Extracted dimensions:")
    print(f"   T_obs: {T_obs}, N: {N}, T_pred: {T_pred}, B: {B}")
    
    # Try the train_unified.py format conversion
    print(f"\\n🔄 Converting to unified format...")
    
    try:
        # Construct input features X_obs: [B, T_obs, N, d_in]
        if args.use_multimodal and args.d_in >= 4:
            X_obs = torch.cat([
                obs_traj.unsqueeze(0),      # [1, T_obs, N, 2] positions
                obs_traj_rel.unsqueeze(0)   # [1, T_obs, N, 2] velocities
            ], dim=-1)  # [B, T_obs, N, 4]
        else:
            X_obs = obs_traj_rel.unsqueeze(0)  # [B, T_obs, N, 2]
        
        print(f"   X_obs: {X_obs.shape}")
        
        # Construct adjacency A_obs: [B, T_obs, N, N]
        print(f"   A_obs raw: {A_obs.shape}")
        A_obs_unified = A_obs[:, 1, :, :, :].permute(0, 1, 2, 3)  # [B, T_obs, N, N]
        print(f"   A_obs_unified: {A_obs_unified.shape}")
        
        # Construct masks
        M_obs = loss_mask[:T_obs].unsqueeze(0)  # [B, T_obs, N]
        M_pred = loss_mask[T_obs:].unsqueeze(0)  # [B, T_pred, N]
        
        print(f"   M_obs: {M_obs.shape}")
        print(f"   M_pred: {M_pred.shape}")
        
        print(f"\\n✅ Unified format conversion successful!")
        
        # Create model
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
        
        print(f"\\n🤖 Model loaded successfully")
        
        # Try forward pass
        print(f"\\n🚀 Testing forward pass...")
        with torch.no_grad():
            delta_Y_pred = model(X_obs, A_obs_unified, M_obs, M_pred=M_pred)
            print(f"   delta_Y_pred: {delta_Y_pred.shape}")
            print(f"\\n✅ Forward pass successful!")
        
    except Exception as e:
        print(f"\\n❌ Error in conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_model_data_interface()