"""
Performance comparison script for integrated model on different datasets
"""

import os
import pickle
import argparse
import torch
import numpy as np
from model import create_integrated_model, multivariate_loss
from utils import TrajectoryDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model on a dataset and return metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Metrics for trajectory prediction
    ade_errors = []  # Average Displacement Error
    fde_errors = []  # Final Displacement Error
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            V_obs, A_obs, V_tr, A_tr = [tensor.to(device) for tensor in batch[-4:]]
            obs_traj, pred_traj_gt = [tensor.to(device) for tensor in batch[:2]]
            
            V_obs_ = V_obs.permute(0, 3, 1, 2)
            V_pred, group_indices = model(V_obs_, A_obs)
            V_pred = V_pred.permute(0, 2, 3, 1)
            
            # Calculate loss
            loss = multivariate_loss(V_pred, V_tr)
            total_loss += loss.item()
            total_samples += 1
            
            # Calculate ADE and FDE
            batch_ade, batch_fde = calculate_ade_fde(V_pred, pred_traj_gt)
            ade_errors.extend(batch_ade)
            fde_errors.extend(batch_fde)
            
            if batch_idx % 100 == 0:
                print(f"Processed batch {batch_idx}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / total_samples
    avg_ade = np.mean(ade_errors)
    avg_fde = np.mean(fde_errors)
    
    return {
        'loss': avg_loss,
        'ade': avg_ade,
        'fde': avg_fde,
        'ade_std': np.std(ade_errors),
        'fde_std': np.std(fde_errors)
    }

def calculate_ade_fde(pred_traj, gt_traj):
    """
    Calculate Average Displacement Error (ADE) and Final Displacement Error (FDE)
    """
    # pred_traj: (batch, num_ped, 2, pred_len)
    # gt_traj: (batch, num_ped, 2, pred_len)
    
    batch_size, num_ped, _, pred_len = pred_traj.shape
    
    ade_errors = []
    fde_errors = []
    
    for b in range(batch_size):
        for p in range(num_ped):
            # Skip if trajectory is empty
            if torch.all(gt_traj[b, p] == 0):
                continue
            
            pred = pred_traj[b, p].cpu().numpy()  # (2, pred_len)
            gt = gt_traj[b, p].cpu().numpy()  # (2, pred_len)
            
            # Calculate displacement errors
            displacement_errors = np.sqrt(np.sum((pred - gt) ** 2, axis=0))
            
            # ADE: average displacement error over all time steps
            ade = np.mean(displacement_errors)
            ade_errors.append(ade)
            
            # FDE: displacement error at final time step
            fde = displacement_errors[-1]
            fde_errors.append(fde)
    
    return ade_errors, fde_errors

def load_model(checkpoint_path, args):
    """
    Load trained model from checkpoint
    """
    model = create_integrated_model(
        n_stgcn=args.n_stgcn, n_tpcnn=args.n_tpcnn, output_feat=args.output_size,
        kernel_size=args.kernel_size, seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len,
        group_d_type=args.group_d_type, group_d_th=args.group_d_th, group_mix_type=args.group_mix_type,
        use_group_processing=args.use_group_processing, density_radius=args.density_radius,
        group_size_threshold=args.group_size_threshold, use_density=args.use_density,
        use_group_size=args.use_group_size
    )
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Model file not found: {checkpoint_path}")
        return None
    
    return model.cuda()

def compare_datasets(model, dataset_configs, device='cuda'):
    """
    Compare model performance on different datasets
    """
    results = {}
    
    for dataset_name, dataset_path in dataset_configs.items():
        print(f"\nEvaluating on {dataset_name}...")
        print(f"Dataset path: {dataset_path}")
        
        # Load dataset
        dataset = TrajectoryDataset(dataset_path, obs_len=8, pred_len=12, skip=1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Evaluate model
        metrics = evaluate_model(model, dataloader, device)
        results[dataset_name] = metrics
        
        print(f"Results for {dataset_name}:")
        print(f"  Loss: {metrics['loss']:.6f}")
        print(f"  ADE: {metrics['ade']:.4f} ± {metrics['ade_std']:.4f}")
        print(f"  FDE: {metrics['fde']:.4f} ± {metrics['fde_std']:.4f}")
    
    return results

def plot_comparison(results, save_path=None):
    """
    Plot comparison results
    """
    datasets = list(results.keys())
    ade_values = [results[dataset]['ade'] for dataset in datasets]
    fde_values = [results[dataset]['fde'] for dataset in datasets]
    ade_stds = [results[dataset]['ade_std'] for dataset in datasets]
    fde_stds = [results[dataset]['fde_std'] for dataset in datasets]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ADE comparison
    x_pos = np.arange(len(datasets))
    bars1 = ax1.bar(x_pos, ade_values, yerr=ade_stds, capsize=5, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('ADE (Average Displacement Error)')
    ax1.set_title('ADE Comparison Across Datasets')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, ade_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    # FDE comparison
    bars2 = ax2.bar(x_pos, fde_values, yerr=fde_stds, capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('FDE (Final Displacement Error)')
    ax2.set_title('FDE Comparison Across Datasets')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, fde_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_tag', default='integrated-dmrgcn-gpgraph', help='Model tag')
    parser.add_argument('--dataset', default='eth', help='Base dataset name')
    parser.add_argument('--checkpoint_path', help='Path to model checkpoint')
    parser.add_argument('--compare_filtered', action='store_true', help='Compare with filtered datasets')
    parser.add_argument('--filtered_base_path', default='./datasets_filtered/', help='Base path for filtered datasets')
    parser.add_argument('--save_plots', action='store_true', help='Save comparison plots')
    
    # Model parameters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcn', type=int, default=1)
    parser.add_argument('--n_tpcnn', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=3)
    
    # GP-Graph parameters
    parser.add_argument('--group_d_type', default='learned_l2norm')
    parser.add_argument('--group_d_th', default='learned')
    parser.add_argument('--group_mix_type', default='mlp')
    parser.add_argument('--use_group_processing', action='store_true', default=True)
    
    # Density/Group size parameters
    parser.add_argument('--density_radius', type=float, default=2.0)
    parser.add_argument('--group_size_threshold', type=int, default=2)
    parser.add_argument('--use_density', action='store_true', default=True)
    parser.add_argument('--use_group_size', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint_path is None:
        checkpoint_dir = f'./checkpoints/{args.model_tag}/'
        checkpoint_path = checkpoint_dir + f'{args.dataset}_best.pth'
    else:
        checkpoint_path = args.checkpoint_path
    
    # Load model
    model = load_model(checkpoint_path, args)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Define datasets to compare
    dataset_configs = {
        'Original Train': f'./datasets/{args.dataset}/train/',
        'Original Val': f'./datasets/{args.dataset}/val/',
        'Original Test': f'./datasets/{args.dataset}/test/',
    }
    
    if args.compare_filtered:
        # Add filtered datasets
        filtered_datasets = [
            'linear_only', 'curved_only', 'direction_change_only',
            'group_motion_only', 'linear_curved', 'all_motions'
        ]
        
        for filtered_name in filtered_datasets:
            filtered_path = os.path.join(args.filtered_base_path, filtered_name)
            if os.path.exists(filtered_path):
                dataset_configs[f'Filtered {filtered_name}'] = filtered_path
    
    # Compare datasets
    print("Comparing model performance across datasets...")
    print("=" * 60)
    
    results = compare_datasets(model, dataset_configs)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        print(f"  ADE: {metrics['ade']:.4f} ± {metrics['ade_std']:.4f}")
        print(f"  FDE: {metrics['fde']:.4f} ± {metrics['fde_std']:.4f}")
        print(f"  Loss: {metrics['loss']:.6f}")
    
    # Plot comparison
    plot_path = f'performance_comparison_{args.model_tag}_{args.dataset}.png' if args.save_plots else None
    plot_comparison(results, plot_path)
    
    # Save results
    results_path = f'performance_results_{args.model_tag}_{args.dataset}.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
