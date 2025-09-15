import os
import pickle
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from model import UnifiedTrajectoryPredictor, ComprehensiveLoss, generate_statistics_matrices
from utils import TrajectoryDataset, compute_metrics, trajectory_visualizer

# Disable cudnn for reproducibility
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

# Argument parsing
parser = argparse.ArgumentParser()

# Model parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_stgcn', type=int, default=1, help='Number of STGCN layers')
parser.add_argument('--n_tpcnn', type=int, default=4, help='Number of CNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

# Data parameters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth', help='Dataset name (eth,hotel,univ,zara1,zara2)')

# Training parameters
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--clip_grad', type=float, default=10.0, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--lr_sh_rate', type=int, default=50, help='LR scheduler step size')
parser.add_argument('--use_lrschd', action="store_true", default=True, help='Use LR scheduler')

# Model-specific parameters
parser.add_argument('--d_type', default='velocity_aware', help='Group detection type')
parser.add_argument('--mix_type', default='attention', help='Group integration type')
parser.add_argument('--include_velocity', action="store_true", default=True)
parser.add_argument('--include_acceleration', action="store_true", default=True)

# Training settings
parser.add_argument('--tag', default='unified-model', help='Experiment tag')
parser.add_argument('--visualize', action="store_true", default=True, help='Visualize trajectories')
parser.add_argument('--save_freq', type=int, default=10, help='Model save frequency')

args = parser.parse_args()

print("Training Unified Trajectory Predictor...")
print(args)

# Setup paths
dataset_path = f'./dataset/{args.dataset}/'
checkpoint_dir = f'./checkpoints/{args.tag}-{args.dataset}/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Save arguments
with open(checkpoint_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)

# Setup tensorboard
writer = SummaryWriter(checkpoint_dir)

# Load datasets
print("Loading datasets...")
train_dataset = TrajectoryDataset(
    dataset_path + 'train/',
    obs_len=args.obs_seq_len,
    pred_len=args.pred_seq_len,
    skip=1,
    include_velocity=args.include_velocity,
    include_acceleration=args.include_acceleration
)

val_dataset = TrajectoryDataset(
    dataset_path + 'val/',
    obs_len=args.obs_seq_len,
    pred_len=args.pred_seq_len,
    skip=1,
    include_velocity=args.include_velocity,
    include_acceleration=args.include_acceleration
)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Setup model
print("Initializing model...")
model = UnifiedTrajectoryPredictor(
    n_stgcn=args.n_stgcn,
    n_tpcnn=args.n_tpcnn,
    input_feat=args.input_size,
    output_feat=args.output_size,
    seq_len=args.obs_seq_len,
    pred_seq_len=args.pred_seq_len,
    kernel_size=args.kernel_size,
    d_type=args.d_type,
    mix_type=args.mix_type,
    group_type=(True, True, True),
    weight_share=True
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# Setup loss function
loss_weights = {
    'prediction': 1.0,
    'group_consistency': 0.1,
    'velocity_consistency': 0.05,
    'social': 0.02,
    'smoothness': 0.01
}
criterion = ComprehensiveLoss(weights=loss_weights).to(device)

# Setup optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Training metrics
metrics = {'train_loss': [], 'val_loss': [], 'train_ade': [], 'val_ade': [], 'train_fde': [], 'val_fde': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}


def train_epoch(epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_losses = {}
    batch_count = 0
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Extract data
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = [t.to(device) for t in batch[:4]]
        V_obs, A_obs, V_pred, A_pred = [t.to(device) for t in batch[-4:]]
        
        optimizer.zero_grad()
        
        # Forward pass
        V_obs_permuted = V_obs.permute(0, 3, 1, 2)  # [batch, channels, time, ped]
        prediction, group_indices, aux_info = model(V_obs_permuted, A_obs)
        
        # Reshape for loss computation
        prediction = prediction.permute(0, 2, 3, 1)  # [batch, time, ped, features]
        V_pred_target = V_pred.permute(0, 2, 3, 1)   # [batch, time, ped, features]
        
        # Compute loss
        total_loss, loss_dict = criterion(
            prediction, V_pred_target[:, :, :, :2],  # Only position for target
            group_indices=group_indices,
            velocity=aux_info.get('velocity'),
            adjacency_matrices=A_obs
        )
        
        # Backward pass
        total_loss.backward()
        
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        
        # Update metrics
        epoch_loss += total_loss.item()
        for key, value in loss_dict.items():
            if key not in epoch_losses:
                epoch_losses[key] = 0.0
            epoch_losses[key] += value.item() if torch.is_tensor(value) else value
        
        batch_count += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Pred': f'{loss_dict.get("prediction", 0):.4f}',
            'Group': f'{loss_dict.get("group_consistency", 0):.4f}'
        })
    
    # Average losses
    epoch_loss /= batch_count
    for key in epoch_losses:
        epoch_losses[key] /= batch_count
    
    return epoch_loss, epoch_losses


def validate_epoch(epoch):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0.0
    epoch_losses = {}
    batch_count = 0
    
    all_predictions = []
    all_targets = []
    all_obs = []
    
    pbar = tqdm(val_loader, desc=f'Val Epoch {epoch}')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Extract data
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = [t.to(device) for t in batch[:4]]
            V_obs, A_obs, V_pred, A_pred = [t.to(device) for t in batch[-4:]]
            
            # Forward pass
            V_obs_permuted = V_obs.permute(0, 3, 1, 2)
            prediction, group_indices, aux_info = model(V_obs_permuted, A_obs)
            
            # Reshape for loss computation
            prediction = prediction.permute(0, 2, 3, 1)
            V_pred_target = V_pred.permute(0, 2, 3, 1)
            
            # Compute loss
            total_loss, loss_dict = criterion(
                prediction, V_pred_target[:, :, :, :2],
                group_indices=group_indices,
                velocity=aux_info.get('velocity'),
                adjacency_matrices=A_obs
            )
            
            # Update metrics
            epoch_loss += total_loss.item()
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item() if torch.is_tensor(value) else value
            
            batch_count += 1
            
            # Store for metrics computation
            all_predictions.append(prediction.cpu())
            all_targets.append(V_pred_target[:, :, :, :2].cpu())
            all_obs.append(obs_traj.cpu())
            
            # Visualize some samples
            if args.visualize and batch_idx < 3:
                # Sample trajectories for visualization
                samples, _ = model.sample_trajectories(V_obs_permuted, A_obs, n_samples=20)
                samples = samples.permute(0, 1, 3, 4, 2)  # [samples, batch, time, ped, 2]
                
                # Convert to absolute coordinates
                last_obs = obs_traj[:, -1:, :, :]
                samples_abs = torch.cumsum(samples, dim=2) + last_obs.unsqueeze(0).cpu()
                pred_abs = torch.cumsum(pred_traj, dim=1) + last_obs.cpu()
                
                # Create visualization
                viz_img = trajectory_visualizer(
                    samples_abs[0, 0], obs_traj[0].cpu(), pred_abs[0].cpu(),
                    group_indices=group_indices.cpu() if group_indices is not None else None
                )
                writer.add_image(f'Val/Sample_{batch_idx}', viz_img, epoch, dataformats='HWC')
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Pred': f'{loss_dict.get("prediction", 0):.4f}'
            })
    
    # Average losses
    epoch_loss /= batch_count
    for key in epoch_losses:
        epoch_losses[key] /= batch_count
    
    return epoch_loss, epoch_losses


def main():
    print("Starting training...")
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_losses = train_epoch(epoch)
        
        # Validate
        val_loss, val_losses = validate_epoch(epoch)
        
        # Update scheduler
        if args.use_lrschd:
            scheduler.step()
        
        # Log metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        for key, value in train_losses.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_losses.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        
        # Print progress
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss'] = val_loss
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir + 'best_model.pth')
            print(f"New best model saved! Val Loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'constant_metrics': constant_metrics
            }, checkpoint_dir + f'checkpoint_epoch_{epoch}.pth')
        
        # Save metrics
        with open(checkpoint_dir + 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)
        
        print(f"Best Val Loss: {constant_metrics['min_val_loss']:.6f} at epoch {constant_metrics['min_val_epoch']}")
        print("-" * 50)
    
    print("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()
