# Unified training script using the new DMRGCN + GP-Graph model
# Uses strict shape conventions and updated data pipeline

import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging

# Import unified model and utilities
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
from datasets.dataloader import TrajectoryDataset, collate_fn
from utils.shapes import set_shape_validation, validate_model_io, log_shape

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable cuDNN for stability
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified DMRGCN + GP-Graph Training')
    
    # Model architecture
    parser.add_argument('--d_in', type=int, default=2, help='Input dimension')
    parser.add_argument('--d_h', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--d_gp_in', type=int, default=128, help='GP-Graph input dimension')
    parser.add_argument('--dmrgcn_hidden_dims', type=int, nargs='+', default=[64, 64, 64, 64, 128],
                       help='DMRGCN hidden dimensions')
    parser.add_argument('--kernel_size', type=int, nargs=2, default=[3, 1],
                       help='(temporal, spatial) kernel size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--distance_scales', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                       help='Distance scales for relations')
    
    # GP-Graph parameters
    parser.add_argument('--agg_method', default='last', choices=['last', 'mean', 'gru'],
                       help='Temporal aggregation method')
    parser.add_argument('--group_type', default='euclidean',
                       choices=['euclidean', 'learned', 'learned_l2norm', 'estimate_th'],
                       help='Group assignment method')
    parser.add_argument('--group_threshold', type=float, default=2.0,
                       help='Group distance threshold')
    parser.add_argument('--mix_type', default='mean',
                       choices=['sum', 'mean', 'mlp', 'cnn', 'attention', 'concat_mlp'],
                       help='Feature fusion method')
    
    # Model configuration
    parser.add_argument('--use_multimodal', action='store_true', default=False,
                       help='Use multi-modal encoder')
    parser.add_argument('--use_simple_head', action='store_true', default=False,
                       help='Use simple regression head instead of GP-Graph')
    parser.add_argument('--enable_agent', action='store_true', default=True,
                       help='Enable agent-level processing')
    parser.add_argument('--enable_intra', action='store_true', default=True,
                       help='Enable intra-group processing')
    parser.add_argument('--enable_inter', action='store_true', default=True,
                       help='Enable inter-group processing')
    
    # Data parameters
    parser.add_argument('--dataset', default='eth',
                       choices=['eth', 'hotel', 'univ', 'zara1', 'zara2'],
                       help='Dataset name')
    parser.add_argument('--obs_len', type=int, default=8, help='Observation length')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction length')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip')
    parser.add_argument('--min_ped', type=int, default=1, help='Min pedestrians per sequence')
    parser.add_argument('--use_cache', action='store_true', default=True, 
                       help='Use cached preprocessed data')
    parser.add_argument('--cache_dir', default='./data_cache', 
                       help='Directory for data cache')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Effective batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--loss_type', default='mse', choices=['mse', 'l1', 'huber'],
                       help='Loss function type')
    
    # Optimization
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--lr_scheduler', default='step', choices=['step', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=25, help='LR step size')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='LR decay factor')
    
    # Logging and checkpointing
    parser.add_argument('--tag', default='unified_dmrgcn_gpgraph', help='Experiment tag')
    parser.add_argument('--checkpoint_dir', default='./checkpoints_unified/', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5, help='Save interval')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    # Debugging
    parser.add_argument('--debug_shapes', action='store_true', default=False,
                       help='Enable detailed shape debugging')
    parser.add_argument('--validate_io', action='store_true', default=True,
                       help='Validate model I/O shapes')
    
    return parser.parse_args()


def setup_model(args):
    """Setup unified model with proper configuration"""
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
    
    return model


def setup_data_loaders(args):
    """Setup data loaders with unified format"""
    dataset_path = f'./copy_dmrgcn/datasets/{args.dataset}/'
    
    train_dataset = TrajectoryDataset(
        dataset_path + 'train/',
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        min_ped=args.min_ped,
        delim='tab',
        use_cache=args.use_cache,
        cache_dir=f'{args.cache_dir}/{args.dataset}'
    )
    
    val_dataset = TrajectoryDataset(
        dataset_path + 'val/',
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        min_ped=args.min_ped,
        delim='tab',
        use_cache=args.use_cache,
        cache_dir=f'{args.cache_dir}/{args.dataset}'
    )
    
    # Use custom collate function for proper batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process one sequence at a time
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def setup_optimizer_scheduler(model, args):
    """Setup optimizer and learning rate scheduler"""
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9
        )
    
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def convert_batch_to_unified_format(batch, device, args):
    """Convert batch to unified [B, T, N, d] format
    
    Returns:
        X_obs: [B, T_obs, N, d_in] - observed features
        A_obs: [B, T_obs, N, N] - observed adjacency
        M_obs: [B, T_obs, N] - observed mask
        delta_Y_true: [B, T_pred, N, 2] - true future deltas
        M_pred: [B, T_pred, N] - prediction mask
    """
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
    # Input: A_obs [B, R, T, N, N] = [1, 2, 8, N, N]
    # Select distance relation (relation index 1) and permute to [B, T, N, N]
    A_obs_unified = A_obs[:, 1, :, :, :].permute(0, 1, 2, 3)  # [B, T_obs, N, N]
    
    # Construct masks
    M_obs = loss_mask[:T_obs].unsqueeze(0)  # [B, T_obs, N]
    M_pred = loss_mask[T_obs:].unsqueeze(0)  # [B, T_pred, N]
    
    # Construct target deltas: [B, T_pred, N, 2]
    delta_Y_true = pred_traj_rel.unsqueeze(0)  # [B, T_pred, N, 2]
    
    return X_obs, A_obs_unified, M_obs, delta_Y_true, M_pred


def train_epoch(model, train_loader, optimizer, device, args):
    """Train for one epoch with unified format"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        # Convert to unified format
        X_obs, A_obs, M_obs, delta_Y_true, M_pred = convert_batch_to_unified_format(
            batch, device, args
        )
        
        if args.debug_shapes:
            log_shape("train_batch", X_obs=X_obs, A_obs=A_obs, M_obs=M_obs, 
                     delta_Y_true=delta_Y_true, M_pred=M_pred)
        
        # Accumulate gradients for effective batch size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()
        
        # Forward pass
        delta_Y_pred = model(X_obs, A_obs, M_obs, M_pred=M_pred)
        
        # Validate I/O if requested
        if args.validate_io:
            input_dict = {'X': X_obs, 'A_obs': A_obs, 'M_obs': M_obs, 'M_pred': M_pred}
            output_dict = {'delta_Y': delta_Y_pred}
            validate_model_io(input_dict, output_dict)
        
        # Compute loss
        loss = model.compute_loss(delta_Y_pred, delta_Y_true, M_pred, args.loss_type)
        loss = loss / args.batch_size  # Scale for accumulation
        
        # Backward pass
        loss.backward()
        
        # Update weights every batch_size iterations
        if (batch_idx + 1) % args.batch_size == 0 or batch_idx == len(train_loader) - 1:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        
        total_loss += loss.item() * args.batch_size
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, device, args):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch in pbar:
            # Convert to unified format
            X_obs, A_obs, M_obs, delta_Y_true, M_pred = convert_batch_to_unified_format(
                batch, device, args
            )
            
            # Forward pass
            delta_Y_pred = model(X_obs, A_obs, M_obs, M_pred=M_pred)
            
            # Compute loss
            loss = model.compute_loss(delta_Y_pred, delta_Y_true, M_pred, args.loss_type)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup shape validation
    set_shape_validation(args.debug_shapes)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Setup directories
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.tag}-{args.dataset}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(checkpoint_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    
    # Setup model
    model = setup_model(args)
    model = model.to(device)
    
    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model Info: {model_info}")
    
    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(args)
    logger.info(f'Train sequences: {len(train_loader)}')
    logger.info(f'Val sequences: {len(val_loader)}')
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_scheduler(model, args)
    
    # Setup tensorboard logging
    writer = SummaryWriter(checkpoint_dir)
    
    # Training metrics
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f'\\nEpoch {epoch}/{args.num_epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device, args)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        logger.info(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % args.save_interval == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'args': args,
                'model_info': model_info
            }
            
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            
            if is_best:
                best_path = os.path.join(checkpoint_dir, f'{args.dataset}_best.pth')
                torch.save(checkpoint, best_path)
                logger.info(f'New best model saved with val loss: {val_loss:.6f}')
    
    writer.close()
    logger.info(f'Training completed. Best val loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()
