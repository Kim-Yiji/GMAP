# Testing script for DMRGCN + GP-Graph integrated model
# Includes evaluation metrics, motion subset analysis, and trajectory visualization

import os
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import our integrated model and dataset
from model.dmrgcn_gpgraph import DMRGCN_GPGraph_Model
from datasets.dataloader import TrajectoryDataset, collate_fn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DMRGCN + GP-Graph Testing')
    
    # Model and data parameters
    parser.add_argument('--dataset', default='eth', 
                       choices=['eth', 'hotel', 'univ', 'zara1', 'zara2'],
                       help='Dataset name')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--obs_len', type=int, default=8, help='Observation sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length')
    
    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=20, help='Number of trajectory samples')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    
    # Analysis parameters
    parser.add_argument('--motion_analysis', action='store_true', default=True,
                       help='Perform motion subset analysis')
    parser.add_argument('--velocity_threshold', type=float, default=0.5, 
                       help='Velocity threshold for motion analysis')
    parser.add_argument('--acceleration_threshold', type=float, default=0.2,
                       help='Acceleration threshold for motion analysis')
    parser.add_argument('--group_size_threshold', type=int, default=3,
                       help='Group size threshold for analysis')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', default=False,
                       help='Enable trajectory visualization')
    parser.add_argument('--save_videos', action='store_true', default=False,
                       help='Save video clips of predictions')
    parser.add_argument('--output_dir', default='./test_outputs/', help='Output directory')
    parser.add_argument('--num_vis_samples', type=int, default=5, 
                       help='Number of samples to visualize')
    
    return parser.parse_args()


def load_model_and_args(checkpoint_path, device):
    """Load model and arguments from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Setup model with same configuration
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, args


def compute_metrics(pred_samples, gt_traj, obs_traj):
    """Compute ADE, FDE, and other metrics
    
    Args:
        pred_samples: (num_samples, pred_len, N, 2) - predicted trajectories
        gt_traj: (pred_len, N, 2) - ground truth trajectories
        obs_traj: (obs_len, N, 2) - observed trajectories
        
    Returns:
        metrics: dict containing ADE, FDE, etc.
    """
    num_samples, pred_len, N, _ = pred_samples.shape
    
    # Convert to absolute coordinates
    last_obs = obs_traj[-1:, :, :]  # (1, N, 2)
    pred_abs = torch.cumsum(pred_samples, dim=1) + last_obs.unsqueeze(0)  # (num_samples, pred_len, N, 2)
    gt_abs = torch.cumsum(gt_traj, dim=0) + last_obs  # (pred_len, N, 2)
    
    # Compute distances for all samples
    distances = torch.norm(pred_abs - gt_abs.unsqueeze(0), p=2, dim=-1)  # (num_samples, pred_len, N)
    
    # Best sample for each pedestrian (minimum final displacement error)
    fde_all = distances[:, -1, :]  # (num_samples, N)
    best_sample_indices = torch.argmin(fde_all, dim=0)  # (N,)
    
    # Compute metrics
    ade_all = distances.mean(dim=1)  # (num_samples, N)
    fde_all = distances[:, -1, :]   # (num_samples, N)
    
    # Best sample metrics
    ade_best = ade_all[best_sample_indices, torch.arange(N)]  # (N,)
    fde_best = fde_all[best_sample_indices, torch.arange(N)]  # (N,)
    
    # Minimum over all samples
    ade_min = ade_all.min(dim=0)[0]  # (N,)
    fde_min = fde_all.min(dim=0)[0]  # (N,)
    
    metrics = {
        'ADE': ade_best.mean().item(),
        'FDE': fde_best.mean().item(),
        'ADE_min': ade_min.mean().item(),
        'FDE_min': fde_min.mean().item(),
        'ADE_per_ped': ade_best.cpu().numpy(),
        'FDE_per_ped': fde_best.cpu().numpy()
    }
    
    return metrics


def analyze_motion_patterns(obs_traj, pred_traj, metrics, args):
    """Analyze metrics by motion patterns
    
    Args:
        obs_traj: (obs_len, N, 2) - observed trajectories  
        pred_traj: (pred_len, N, 2) - ground truth future trajectories
        metrics: dict with per-pedestrian metrics
        args: command line arguments
        
    Returns:
        motion_analysis: dict with analysis results
    """
    obs_len, N, _ = obs_traj.shape
    
    # Compute velocities and accelerations
    velocities = torch.norm(obs_traj[1:] - obs_traj[:-1], p=2, dim=-1)  # (obs_len-1, N)
    avg_velocity = velocities.mean(dim=0)  # (N,)
    
    if obs_len > 2:
        accelerations = torch.norm(velocities[1:] - velocities[:-1], p=2, dim=-1)  # (obs_len-2, N)
        avg_acceleration = accelerations.mean(dim=0)  # (N,)
    else:
        avg_acceleration = torch.zeros(N)
    
    # Classify pedestrians
    high_velocity_mask = avg_velocity > args.velocity_threshold
    high_accel_mask = avg_acceleration > args.acceleration_threshold
    
    # Create motion categories
    categories = {
        'static': (~high_velocity_mask) & (~high_accel_mask),
        'linear': high_velocity_mask & (~high_accel_mask),
        'non_linear': high_velocity_mask & high_accel_mask,
        'accelerating': (~high_velocity_mask) & high_accel_mask
    }
    
    # Compute metrics per category
    motion_analysis = {}
    for category, mask in categories.items():
        if mask.sum() > 0:
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            ade_cat = np.array(metrics['ADE_per_ped'])[indices.cpu().numpy()]
            fde_cat = np.array(metrics['FDE_per_ped'])[indices.cpu().numpy()]
            
            motion_analysis[category] = {
                'count': len(indices),
                'ADE': ade_cat.mean(),
                'FDE': fde_cat.mean(),
                'avg_velocity': avg_velocity[indices].mean().item(),
                'avg_acceleration': avg_acceleration[indices].mean().item()
            }
        else:
            motion_analysis[category] = {
                'count': 0,
                'ADE': 0.0,
                'FDE': 0.0,
                'avg_velocity': 0.0,
                'avg_acceleration': 0.0
            }
    
    return motion_analysis


def visualize_predictions(obs_traj, pred_samples, gt_traj, group_indices, save_path=None):
    """Visualize trajectory predictions
    
    Args:
        obs_traj: (obs_len, N, 2) - observed trajectories
        pred_samples: (num_samples, pred_len, N, 2) - predicted trajectories  
        gt_traj: (pred_len, N, 2) - ground truth trajectories
        group_indices: (N,) - group assignments
        save_path: path to save figure
    """
    obs_len, N, _ = obs_traj.shape
    num_samples, pred_len, _, _ = pred_samples.shape
    
    # Convert to absolute coordinates
    last_obs = obs_traj[-1:, :, :]
    obs_abs = torch.cumsum(obs_traj, dim=0)
    pred_abs = torch.cumsum(pred_samples, dim=2) + last_obs.unsqueeze(0).unsqueeze(0)
    gt_abs = torch.cumsum(gt_traj, dim=0) + last_obs
    
    # Convert to numpy
    obs_abs = obs_abs.cpu().numpy()
    pred_abs = pred_abs.cpu().numpy()
    gt_abs = gt_abs.cpu().numpy()
    group_indices = group_indices.cpu().numpy()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Color map for groups
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(group_indices))))
    
    for n in range(N):
        group_id = group_indices[n]
        color = colors[group_id]
        
        # Plot observed trajectory
        plt.plot(obs_abs[:, n, 0], obs_abs[:, n, 1], 'o-', color=color, 
                linewidth=2, markersize=4, label=f'Obs {n}' if n < 3 else "")
        
        # Plot ground truth
        plt.plot(gt_abs[:, n, 0], gt_abs[:, n, 1], 's-', color=color,
                linewidth=2, markersize=4, alpha=0.7, label=f'GT {n}' if n < 3 else "")
        
        # Plot predictions (sample a few)
        for s in range(min(5, num_samples)):
            alpha = 0.3 if s > 0 else 0.6
            plt.plot(pred_abs[s, :, n, 0], pred_abs[s, :, n, 1], '--', 
                    color=color, alpha=alpha, linewidth=1,
                    label=f'Pred {n}' if s == 0 and n < 3 else "")
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trajectory Predictions with Group Assignments')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Main testing function"""
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, model_args = load_model_and_args(args.checkpoint, device)
    print(f'Loaded model from {args.checkpoint}')
    
    # Setup data loader
    dataset_path = f'./copy_dmrgcn/datasets/{args.dataset}/'
    test_dataset = TrajectoryDataset(
        dataset_path + 'test/',
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=1,
        min_ped=1,
        delim='tab'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f'Test sequences: {len(test_loader)}')
    
    # Testing
    all_metrics = []
    all_motion_analysis = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
             non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
             seq_start_end, agent_ids) = batch
            
            # Move to device
            V_obs = V_obs.to(device).float()
            A_obs = A_obs.to(device).float()
            obs_traj = obs_traj.to(device).float()
            pred_traj = pred_traj.to(device).float()
            
            # Forward pass
            predictions, group_indices = model(V_obs, A_obs)
            
            # Sample trajectories
            pred_samples = model.predict_trajectories(predictions, args.num_samples)
            
            # Convert format: (num_samples, B, T, N, 2) -> (num_samples, T, N, 2)
            pred_samples = pred_samples.squeeze(1)  # Remove batch dimension
            
            # Compute metrics
            metrics = compute_metrics(pred_samples, pred_traj, obs_traj)
            all_metrics.append(metrics)
            
            # Motion analysis
            if args.motion_analysis:
                motion_analysis = analyze_motion_patterns(obs_traj, pred_traj, metrics, args)
                all_motion_analysis.append(motion_analysis)
            
            # Visualization
            if args.visualize and batch_idx < args.num_vis_samples:
                save_path = os.path.join(args.output_dir, f'prediction_{batch_idx}.png')
                visualize_predictions(
                    obs_traj, pred_samples, pred_traj, 
                    group_indices.squeeze(0), save_path
                )
            
            # Update progress
            pbar.set_postfix({
                'ADE': f'{metrics["ADE"]:.4f}',
                'FDE': f'{metrics["FDE"]:.4f}'
            })
    
    # Aggregate results
    print("\\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Overall metrics
    overall_ade = np.mean([m['ADE'] for m in all_metrics])
    overall_fde = np.mean([m['FDE'] for m in all_metrics])
    overall_ade_min = np.mean([m['ADE_min'] for m in all_metrics])
    overall_fde_min = np.mean([m['FDE_min'] for m in all_metrics])
    
    print(f"Overall ADE: {overall_ade:.4f}")
    print(f"Overall FDE: {overall_fde:.4f}")
    print(f"Overall ADE (min): {overall_ade_min:.4f}")
    print(f"Overall FDE (min): {overall_fde_min:.4f}")
    
    # Motion subset analysis
    if args.motion_analysis and all_motion_analysis:
        print("\\nMOTION SUBSET ANALYSIS:")
        print("-" * 30)
        
        # Aggregate motion analysis
        categories = ['static', 'linear', 'non_linear', 'accelerating']
        for category in categories:
            cat_metrics = [ma[category] for ma in all_motion_analysis if ma[category]['count'] > 0]
            if cat_metrics:
                avg_ade = np.mean([m['ADE'] for m in cat_metrics])
                avg_fde = np.mean([m['FDE'] for m in cat_metrics])
                total_count = sum([m['count'] for m in cat_metrics])
                
                print(f"{category.upper()}:")
                print(f"  Count: {total_count}")
                print(f"  ADE: {avg_ade:.4f}")
                print(f"  FDE: {avg_fde:.4f}")
    
    # Save results
    results = {
        'overall_metrics': {
            'ADE': overall_ade,
            'FDE': overall_fde,
            'ADE_min': overall_ade_min,
            'FDE_min': overall_fde_min
        },
        'all_metrics': all_metrics,
        'motion_analysis': all_motion_analysis,
        'args': args,
        'model_args': model_args
    }
    
    results_path = os.path.join(args.output_dir, f'results_{args.dataset}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\\nResults saved to {results_path}")
    print("Testing completed!")


if __name__ == '__main__':
    main()
