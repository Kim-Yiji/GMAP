# Training script for DMRGCN + GP-Graph integrated model
# Enhanced with group-aware trajectory prediction capabilities

import os
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import our integrated model and dataset
from model.dmrgcn_gpgraph import DMRGCNGPGraph
from datasets.dataloader import TrajectoryDataset, collate_fn

# Avoid contiguous problems
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DMRGCN + GP-Graph Training')
    
    # Model architecture parameters
    parser.add_argument('--input_dim', type=int, default=2, help='Input dimension (x, y)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64, 64, 64, 5], 
                       help='Hidden dimensions for DMRGCN backbone')
    parser.add_argument('--kernel_size', type=int, nargs=2, default=[3, 1], 
                       help='Temporal and spatial kernel sizes')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Group-aware parameters
    parser.add_argument('--gpgraph', action='store_true', default=True, 
                       help='Enable GP-Graph group-aware processing')
    parser.add_argument('--group_type', default='euclidean', 
                       choices=['euclidean', 'learned', 'learned_l2norm', 'estimate_th'],
                       help='Group assignment method')
    parser.add_argument('--group_th', type=float, default=2.0, help='Group distance threshold')
    parser.add_argument('--mix_type', default='mean', 
                       choices=['sum', 'mean', 'mlp', 'cnn', 'attention', 'concat_mlp'],
                       help='Feature fusion method')
    parser.add_argument('--enable_agent', action='store_true', default=True,
                       help='Enable agent-level processing')
    parser.add_argument('--enable_intra', action='store_true', default=True,
                       help='Enable intra-group processing')
    parser.add_argument('--enable_inter', action='store_true', default=True,
                       help='Enable inter-group processing')
    parser.add_argument('--share_backbone', action='store_true', default=True,
                       help='Share DMRGCN backbone across paths')
    parser.add_argument('--st_estimator', action='store_true', default=False,
                       help='Use spatio-temporal features for group estimation')
    
    # DMRGCN-specific parameters
    parser.add_argument('--distance_scales', type=float, nargs='+', default=[0.5, 1.0, 2.0],
                       help='Distance scales for multi-relational graphs')
    parser.add_argument('--use_mdn', action='store_true', default=True,
                       help='Use mixture density network for predictions')
    
    # Data parameters
    parser.add_argument('--dataset', default='eth', 
                       choices=['eth', 'hotel', 'univ', 'zara1', 'zara2'],
                       help='Dataset name')
    parser.add_argument('--obs_len', type=int, default=8, help='Observation sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip for data loading')
    parser.add_argument('--min_ped', type=int, default=1, help='Minimum pedestrians per sequence')
    parser.add_argument('--delim', default='tab', help='Data delimiter')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=128, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--lr_scheduler', default='step', choices=['step', 'cosine', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=32, help='Step size for lr scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='Gamma for lr scheduler')
    
    # Logging and checkpointing
    parser.add_argument('--tag', default='dmrgcn_gpgraph', help='Experiment tag')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/', help='Checkpoint directory')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    
    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples for evaluation')
    parser.add_argument('--visualize', action='store_true', default=False, 
                       help='Enable trajectory visualization')
    
    return parser.parse_args()


def setup_model(args):
    """Setup the integrated model"""
    enable_paths = {
        'agent': args.enable_agent,
        'intra': args.enable_intra, 
        'inter': args.enable_inter
    }
    
    model = DMRGCNGPGraph(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        kernel_size=tuple(args.kernel_size),
        dropout=args.dropout,
        group_type=args.group_type,
        group_threshold=args.group_th,
        mix_type=args.mix_type,
        enable_paths=enable_paths,
        distance_scales=args.distance_scales,
        share_backbone=args.share_backbone,
        use_mdn=args.use_mdn,
        st_estimator=args.st_estimator
    )
    
    return model


def setup_data_loaders(args):
    """Setup data loaders"""
    dataset_path = f'./copy_dmrgcn/datasets/{args.dataset}/'
    
    train_dataset = TrajectoryDataset(
        dataset_path + 'train/',
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        min_ped=args.min_ped,
        delim=args.delim
    )
    
    val_dataset = TrajectoryDataset(
        dataset_path + 'val/',
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        min_ped=args.min_ped,
        delim=args.delim
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process one sequence at a time due to variable sizes
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
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


def train_epoch(model, train_loader, optimizer, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        # Unpack batch
        (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, 
         non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred, 
         seq_start_end, agent_ids) = batch
        
        # Move to device
        V_obs = V_obs.to(device).float()
        A_obs = A_obs.to(device).float()
        pred_traj = pred_traj.to(device).float()
        loss_mask = loss_mask.to(device).float()
        
        # Accumulate gradients for effective batch size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()
        
        # Forward pass
        predictions, group_indices = model(V_obs, A_obs)
        
        # Compute loss
        # Convert predictions and targets to proper format
        # predictions: (B, 5, pred_len, N)
        # pred_traj: (pred_len, N, 2) -> need to reshape
        
        B, _, pred_len, N = predictions.shape
        targets = pred_traj.permute(1, 0, 2).unsqueeze(0)  # (1, N, pred_len, 2)
        targets = targets.permute(0, 2, 1, 3)  # (1, pred_len, N, 2)
        
        # Create mask for prediction length
        pred_mask = loss_mask[-pred_len:, :].unsqueeze(0)  # (1, pred_len, N)
        
        loss = model.compute_loss(predictions, targets, pred_mask)
        loss = loss / args.batch_size  # Scale loss for accumulation
        
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
            # Unpack batch
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
             seq_start_end, agent_ids) = batch
            
            # Move to device
            V_obs = V_obs.to(device).float()
            A_obs = A_obs.to(device).float()
            pred_traj = pred_traj.to(device).float()
            loss_mask = loss_mask.to(device).float()
            
            # Forward pass
            predictions, group_indices = model(V_obs, A_obs)
            
            # Compute loss
            B, _, pred_len, N = predictions.shape
            targets = pred_traj.permute(1, 0, 2).unsqueeze(0)
            targets = targets.permute(0, 2, 1, 3)
            pred_mask = loss_mask[-pred_len:, :].unsqueeze(0)
            
            loss = model.compute_loss(predictions, targets, pred_mask)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    return total_loss / num_batches


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup directories
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.tag}-{args.dataset}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(checkpoint_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    
    # Setup model
    model = setup_model(args)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(args)
    print(f'Train sequences: {len(train_loader)}')
    print(f'Val sequences: {len(val_loader)}')
    
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
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\\nEpoch {epoch}/{args.num_epochs}')
        
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
        
        print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
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
                'args': args
            }
            
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            
            if is_best:
                best_path = os.path.join(checkpoint_dir, f'{args.dataset}_best.pth')
                torch.save(checkpoint, best_path)
                print(f'New best model saved with val loss: {val_loss:.6f}')
    
    writer.close()
    print(f'Training completed. Best val loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()
