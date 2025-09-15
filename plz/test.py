import os
import pickle
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import UnifiedTrajectoryPredictor, generate_statistics_matrices
from utils import TrajectoryDataset, compute_metrics, compute_group_metrics, trajectory_visualizer

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='unified-model', help='Experiment tag')
parser.add_argument('--dataset', default='eth', help='Dataset to test on')
parser.add_argument('--n_samples', type=int, default=20, help='Number of samples for evaluation')
parser.add_argument('--n_trials', type=int, default=100, help='Number of evaluation trials')
parser.add_argument('--visualize', action="store_true", default=True, help='Generate visualizations')
parser.add_argument('--model_path', default=None, help='Specific model path (default: best_model.pth)')

test_args = parser.parse_args()

print("Testing Unified Trajectory Predictor...")
print(test_args)

# Setup paths
checkpoint_dir = f'./checkpoints/{test_args.tag}-{test_args.dataset}/'
model_path = test_args.model_path if test_args.model_path else checkpoint_dir + 'best_model.pth'

# Load training arguments
args_path = checkpoint_dir + 'args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)

# Override dataset if specified
if test_args.dataset != args.dataset:
    print(f"Overriding dataset from {args.dataset} to {test_args.dataset}")
    args.dataset = test_args.dataset

dataset_path = f'./dataset/{args.dataset}/'

print(f"Loading model from: {model_path}")
print(f"Testing on dataset: {args.dataset}")

# Load test dataset
print("Loading test dataset...")
test_dataset = TrajectoryDataset(
    dataset_path + 'test/',
    obs_len=args.obs_seq_len,
    pred_len=args.pred_seq_len,
    skip=1,
    include_velocity=getattr(args, 'include_velocity', True),
    include_acceleration=getattr(args, 'include_acceleration', True)
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"Test samples: {len(test_dataset)}")

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
    d_type=getattr(args, 'd_type', 'velocity_aware'),
    mix_type=getattr(args, 'mix_type', 'attention'),
    group_type=(True, True, True),
    weight_share=True
).cuda()

# Load model weights
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")
else:
    print(f"Model file not found: {model_path}")
    exit(1)

# Setup tensorboard for test visualization
writer = SummaryWriter(checkpoint_dir + 'test_results/')


@torch.no_grad()
def test_model():
    """Test the model and compute comprehensive metrics"""
    model.eval()
    
    all_ade = []
    all_fde = []
    all_col = []
    all_tcc = []
    
    group_metrics_list = []
    velocity_metrics_list = []
    
    sample_visualizations = []
    
    print("Running inference...")
    pbar = tqdm(test_loader, desc='Testing')
    
    for batch_idx, batch in enumerate(pbar):
        # Extract data
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = [t.cuda() for t in batch[:4]]
        V_obs, A_obs, V_pred, A_pred = [t.cuda() for t in batch[-4:]]
        
        # Convert to absolute coordinates for evaluation
        obs_traj_abs = obs_traj
        pred_traj_abs = torch.cumsum(pred_traj_rel, dim=1) + obs_traj[:, -1:, :, :]
        
        # Multiple evaluation trials to reduce randomness
        trial_metrics = {'ADE': [], 'FDE': [], 'COL': [], 'TCC': []}
        
        for trial in range(test_args.n_trials):
            # Forward pass
            V_obs_permuted = V_obs.permute(0, 3, 1, 2)
            prediction, group_indices, aux_info = model(V_obs_permuted, A_obs)
            
            # Sample trajectories
            pred_samples, _ = model.sample_trajectories(
                V_obs_permuted, A_obs, 
                n_samples=test_args.n_samples, 
                use_group_sampling=True
            )
            
            # Convert samples to absolute coordinates
            pred_samples = pred_samples.permute(0, 1, 3, 4, 2)  # [samples, batch, time, ped, 2]
            last_obs = obs_traj[:, -1:, :, :]
            pred_samples_abs = torch.cumsum(pred_samples, dim=2) + last_obs.unsqueeze(0).cpu()
            
            # Compute metrics for this trial
            metrics = compute_metrics(pred_samples_abs, pred_traj_abs.cpu(), obs_traj.cpu())
            
            trial_metrics['ADE'].append(metrics['ADE'])
            trial_metrics['FDE'].append(metrics['FDE'])
            trial_metrics['COL'].append(metrics['COL'])
            trial_metrics['TCC'].append(metrics['TCC'])
        
        # Average over trials
        ade = np.mean(trial_metrics['ADE'])
        fde = np.mean(trial_metrics['FDE'])
        col = np.mean(trial_metrics['COL'])
        tcc = np.mean(trial_metrics['TCC'])
        
        all_ade.append(ade)
        all_fde.append(fde)
        all_col.append(col)
        all_tcc.append(tcc)
        
        # Compute group-specific metrics
        if group_indices is not None:
            group_metrics = compute_group_metrics(
                pred_samples_abs, pred_traj_abs.cpu(), group_indices.cpu()
            )
            group_metrics_list.append(group_metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'ADE': f'{ade:.4f}',
            'FDE': f'{fde:.4f}',
            'COL': f'{col:.2f}',
            'TCC': f'{tcc:.4f}'
        })
        
        # Visualize some samples
        if test_args.visualize and batch_idx < 10:
            # Use the last trial's predictions for visualization
            viz_img = trajectory_visualizer(
                pred_samples_abs[:, 0], obs_traj[0].cpu(), pred_traj_abs[0].cpu(),
                group_indices=group_indices.cpu() if group_indices is not None else None,
                samples=min(10, test_args.n_samples)
            )
            writer.add_image(f'Test/Sample_{batch_idx}', viz_img, 0, dataformats='HWC')
            
            if batch_idx < 3:
                sample_visualizations.append({
                    'pred_samples': pred_samples_abs[:5, 0].numpy(),  # Save only 5 samples
                    'obs_traj': obs_traj[0].cpu().numpy(),
                    'pred_traj': pred_traj_abs[0].cpu().numpy(),
                    'group_indices': group_indices.cpu().numpy() if group_indices is not None else None
                })
    
    # Compute final metrics
    final_metrics = {
        'ADE': np.mean(all_ade),
        'FDE': np.mean(all_fde),
        'COL': np.mean(all_col),
        'TCC': np.mean(all_tcc),
        'ADE_std': np.std(all_ade),
        'FDE_std': np.std(all_fde),
        'COL_std': np.std(all_col),
        'TCC_std': np.std(all_tcc)
    }
    
    # Compute group metrics
    if group_metrics_list:
        avg_group_metrics = {}
        for key in group_metrics_list[0].keys():
            values = [gm[key] for gm in group_metrics_list if key in gm]
            if values:
                avg_group_metrics[f'group_{key}'] = np.mean(values)
        final_metrics.update(avg_group_metrics)
    
    return final_metrics, sample_visualizations


def save_results(metrics, sample_visualizations):
    """Save test results"""
    # Save metrics
    results_path = checkpoint_dir + 'test_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save sample visualizations
    if sample_visualizations:
        viz_path = checkpoint_dir + 'sample_visualizations.pkl'
        with open(viz_path, 'wb') as f:
            pickle.dump(sample_visualizations, f)
    
    # Save readable results
    results_txt_path = checkpoint_dir + 'test_results.txt'
    with open(results_txt_path, 'w') as f:
        f.write(f"Test Results for {test_args.tag} on {args.dataset}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Samples per prediction: {test_args.n_samples}\n")
        f.write(f"Evaluation trials: {test_args.n_trials}\n")
        f.write("\nMetrics:\n")
        f.write("-" * 30 + "\n")
        
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"Results saved to {results_txt_path}")


def print_results(metrics):
    """Print test results in a formatted way"""
    print("\n" + "=" * 60)
    print(f"TEST RESULTS - {test_args.tag} on {args.dataset}")
    print("=" * 60)
    
    # Main metrics
    print(f"ADE: {metrics['ADE']:.4f} ± {metrics['ADE_std']:.4f}")
    print(f"FDE: {metrics['FDE']:.4f} ± {metrics['FDE_std']:.4f}")
    print(f"COL: {metrics['COL']:.2f} ± {metrics['COL_std']:.2f}")
    print(f"TCC: {metrics['TCC']:.4f} ± {metrics['TCC_std']:.4f}")
    
    # Group metrics if available
    group_keys = [k for k in metrics.keys() if k.startswith('group_')]
    if group_keys:
        print("\nGroup Metrics:")
        print("-" * 20)
        for key in group_keys:
            print(f"{key}: {metrics[key]:.4f}")
    
    print("=" * 60)


def main():
    """Main testing function"""
    print("Starting evaluation...")
    
    # Run test
    metrics, sample_visualizations = test_model()
    
    # Print results
    print_results(metrics)
    
    # Save results
    save_results(metrics, sample_visualizations)
    
    # Log final metrics to tensorboard
    for key, value in metrics.items():
        writer.add_scalar(f'Final/{key}', value, 0)
    
    writer.close()
    print("Testing completed!")


if __name__ == "__main__":
    main()
