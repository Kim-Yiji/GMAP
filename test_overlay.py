# Testing script for DMRGCN + GP-Graph integrated model
# Includes evaluation metrics, motion subset analysis, and trajectory visualization

import os
import pickle
import argparse
from torch.serialization import add_safe_globals
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2

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
    parser.add_argument('--motion_analysis', action='store_true', default=False,
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
    parser.add_argument('--video_path', type=str, default=None,
                       help='Optional: path to input video for overlay')
    parser.add_argument('--homography_path', type=str, default=None,
                       help='Optional: path to 3x3 homography txt for world->pixel mapping')
    parser.add_argument('--fps', type=int, default=25, help='Video FPS when writing output')
    parser.add_argument('--start_frame', type=int, default=65,
                       help='Start frame index to align dataset sample to video')
    parser.add_argument('--ann_step', type=int, default=1,
                       help='Frames between successive annotations (1 if annotated every video frame)')
    
    return parser.parse_args()


def load_model_and_args(checkpoint_path, device):
    """Load model and arguments from checkpoint"""
    # Allowlist argparse.Namespace for PyTorch 2.6+ safe load
    add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']

    # Map checkpoint args -> unified model config
    enable_paths = {
        'agent': getattr(args, 'enable_agent', True),
        'intra': getattr(args, 'enable_intra', True),
        'inter': getattr(args, 'enable_inter', True)
    }

    model = DMRGCN_GPGraph_Model(
        d_in=getattr(args, 'd_in', 2),
        d_h=getattr(args, 'd_h', 128),
        d_gp_in=getattr(args, 'd_gp_in', 128),
        T_pred=getattr(args, 'pred_len', 12),
        output_dim=2,
        dmrgcn_hidden_dims=getattr(args, 'dmrgcn_hidden_dims', [64, 64, 64, 64, 128]),
        dmrgcn_kernel_size=tuple(getattr(args, 'kernel_size', [3, 1])),
        dmrgcn_dropout=getattr(args, 'dropout', 0.1),
        distance_scales=getattr(args, 'distance_scales', [0.5, 1.0, 2.0]),
        agg_method=getattr(args, 'agg_method', 'last'),
        group_type=getattr(args, 'group_type', 'euclidean'),
        group_threshold=getattr(args, 'group_threshold', 2.0),
        mix_type=getattr(args, 'mix_type', 'mean'),
        enable_paths=enable_paths,
        use_multimodal=getattr(args, 'use_multimodal', False),
        use_simple_head=getattr(args, 'use_simple_head', False),
        share_backbone=getattr(args, 'share_backbone', True)
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

    # Move to CPU and ensure 1-D
    avg_velocity = avg_velocity.detach().cpu().reshape(-1)
    avg_acceleration = avg_acceleration.detach().cpu().reshape(-1)
    
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
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1).cpu()
            if indices.numel() == 0:
                motion_analysis[category] = {
                    'count': 0,
                    'ADE': 0.0,
                    'FDE': 0.0,
                    'avg_velocity': 0.0,
                    'avg_acceleration': 0.0
                }
            else:
                ade_cat = np.array(metrics['ADE_per_ped'])[indices.cpu().numpy()]
                fde_cat = np.array(metrics['FDE_per_ped'])[indices.cpu().numpy()]
                motion_analysis[category] = {
                    'count': len(indices),
                    'ADE': float(ade_cat.mean()) if ade_cat.size else 0.0,
                    'FDE': float(fde_cat.mean()) if fde_cat.size else 0.0,
                    'avg_velocity': avg_velocity[indices].mean().item() if indices.numel() else 0.0,
                    'avg_acceleration': avg_acceleration[indices].mean().item() if indices.numel() else 0.0
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


# ============= Video overlay utilities (OpenCV) =============
def load_homography(h_path):
    H = np.loadtxt(h_path)
    if H.shape != (3, 3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H


def world_to_pixel(points_xy, H):
    # points_xy: (K, 2)
    if points_xy.size == 0:
        return points_xy
    ones = np.ones((points_xy.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([points_xy.astype(np.float64), ones])  # (K,3)
    proj = (H @ pts_h.T).T                                    # (K,3)
    uv = proj[:, :2] / proj[:, 2:3]
    return uv.astype(np.float32)


def draw_trajectories_on_frame(frame, obs_abs, pred_abs, gt_abs,
                               color_obs=(0, 200, 255),
                               color_pred=(0, 255, 0),
                               color_gt=(255, 0, 0)):
    """Draw trajectories on a BGR frame.
    obs_abs: (T_obs, N, 2) absolute coords in pixel
    pred_abs: (T_pred, N, 2) absolute coords in pixel
    gt_abs: (T_pred, N, 2) absolute coords in pixel
    """
    h, w = frame.shape[:2]

    def to_int_pts(seq):
        if seq.size == 0:
            return []
        pts = np.round(seq).astype(np.int32)
        # clip inside frame
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts

    # draw per-agent polylines
    T_obs, N, _ = obs_abs.shape
    T_pred = pred_abs.shape[0]

    for n in range(N):
        # observed
        obs_pts = to_int_pts(obs_abs[:, n, :])
        if len(obs_pts) >= 2:
            cv2.polylines(frame, [obs_pts.reshape(-1, 1, 2)], False, color_obs, 2, cv2.LINE_AA)
        for p in obs_pts:
            cv2.circle(frame, tuple(p), 2, color_obs, -1, cv2.LINE_AA)

        # ground truth future
        gt_pts = to_int_pts(gt_abs[:, n, :])
        if len(gt_pts) >= 2:
            cv2.polylines(frame, [gt_pts.reshape(-1, 1, 2)], False, color_gt, 2, cv2.LINE_AA)

        # predicted future (first sample)
        pred_pts = to_int_pts(pred_abs[:, n, :])
        if len(pred_pts) >= 2:
            cv2.polylines(frame, [pred_pts.reshape(-1, 1, 2)], False, color_pred, 2, cv2.LINE_AA)

        # agent id near last point (if available)
        anchor = pred_pts[-1] if len(pred_pts) else (obs_pts[-1] if len(obs_pts) else None)
        if anchor is not None:
            cv2.putText(frame, f"ID{n}", tuple(anchor + np.array([3, -3])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def convert_batch_to_unified_format_for_inference(batch, device, use_multimodal=False, d_in=2):
    """Prepare inputs for unified model inference (mirror of train_unified.convert_batch_to_unified_format)."""
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     non_linear_ped, loss_mask, V_obs, A_obs, V_pred, A_pred,
     seq_start_end, agent_ids) = batch

    # Move to device
    obs_traj = obs_traj.to(device).float()
    pred_traj = pred_traj.to(device).float()
    obs_traj_rel = obs_traj_rel.to(device).float()
    pred_traj_rel = pred_traj_rel.to(device).float()
    A_obs = A_obs.to(device).float()
    loss_mask = loss_mask.to(device).float()

    T_obs, N = obs_traj.shape[:2]
    # Build X_obs
    if use_multimodal and d_in >= 4:
        X_obs = torch.cat([obs_traj.unsqueeze(0), obs_traj_rel.unsqueeze(0)], dim=-1)
    else:
        X_obs = obs_traj_rel.unsqueeze(0)

    # Adjacency: select distance relation (index 1), shape -> [B, T, N, N]
    A_obs_unified = A_obs[:, 1, :, :, :].permute(0, 1, 2, 3)

    # Masks
    M_obs = loss_mask[:T_obs].unsqueeze(0)
    M_pred = loss_mask[T_obs:].unsqueeze(0)

    return X_obs, A_obs_unified, M_obs, M_pred, obs_traj, pred_traj


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
    
    # Prepare video IO if requested
    cap = None
    writer = None
    H = None
    if args.video_path is not None and (args.save_videos or args.visualize):
        if args.homography_path is not None and os.path.isfile(args.homography_path):
            H = load_homography(args.homography_path)
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f'[WARN] Failed to open video: {args.video_path}')
            cap = None
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            fps = args.fps if args.fps else (fps_in if fps_in and fps_in > 0 else 25)
            out_path = os.path.join(args.output_dir, f'overlay_{args.dataset}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
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
            
            # Prepare unified inputs
            X_obs, A_obs_u, M_obs, M_pred, obs_abs_ref, gt_rel = convert_batch_to_unified_format_for_inference(
                batch, device, use_multimodal=model_args.use_multimodal if hasattr(model_args, 'use_multimodal') else False,
                d_in=model_args.d_in if hasattr(model_args, 'd_in') else 2
            )

            # Predict deltas then compose absolute with last absolute obs
            delta_Y = model(X_obs, A_obs_u, M_obs, M_pred=M_pred)  # [B, T_pred, N, 2]
            delta_Y = delta_Y.squeeze(0)  # [T_pred, N, 2]
            last_obs_abs = obs_abs_ref[-1:, :, :]  # [1, N, 2] absolute
            pred_abs = torch.cumsum(delta_Y, dim=0) + last_obs_abs  # [T_pred, N, 2]
            
            # Absolute obs/gt for overlay
            obs_abs = torch.cumsum(obs_abs_ref, dim=0)  # [T_obs, N, 2]
            gt_abs = torch.cumsum(gt_rel, dim=0) + obs_abs[-1:, :, :]  # [T_pred, N, 2]
            
            # Compute metrics (use single-sample absolute pred)
            # Convert to expected shapes
            pred_for_metrics = (pred_abs.unsqueeze(0))  # [1, T_pred, N, 2]
            metrics = compute_metrics(pred_for_metrics, gt_rel, obs_abs_ref)
            all_metrics.append(metrics)
            
            # Motion analysis
            if args.motion_analysis:
                motion_analysis = analyze_motion_patterns(obs_traj, pred_traj, metrics, args)
                all_motion_analysis.append(motion_analysis)
            
            # Visualization (optional): skip group indices in unified path
            if args.visualize and batch_idx < args.num_vis_samples:
                save_path = os.path.join(args.output_dir, f'prediction_{batch_idx}.png')
                # Reuse existing function by constructing minimal inputs
                visualize_predictions(
                    obs_abs_ref, pred_for_metrics, gt_rel, 
                    torch.zeros(obs_abs_ref.shape[1], dtype=torch.long), save_path
                )
            
            # Video overlay with per-frame interpolation to match video FPS
            if cap is not None and writer is not None:
                # Use computed absolute coords
                obs_abs_np = obs_abs.detach().cpu().numpy()
                pred_abs_np = pred_abs.detach().cpu().numpy()
                gt_abs_np = gt_abs.detach().cpu().numpy()

                # Map to pixel if homography is provided
                if H is not None:
                    def map_seq(seq):
                        t, n, _ = seq.shape
                        seq_flat = seq.reshape(-1, 2)
                        uv = world_to_pixel(seq_flat, H)
                        return uv.reshape(t, n, 2)
                    obs_abs_px = map_seq(obs_abs_np)
                    pred_abs_px = map_seq(pred_abs_np)
                    gt_abs_px = map_seq(gt_abs_np)
                else:
                    obs_abs_px = obs_abs_np
                    pred_abs_px = pred_abs_np
                    gt_abs_px = gt_abs_np

                # Determine annotation step in video frames (ann_step=1 if per-frame)
                step = max(1, int(args.ann_step))

                # Build a short clip covering this sequence time span
                # Obs covers T_obs timesteps, pred covers T_pred timesteps → total_ann = T_obs+T_pred
                T_obs = obs_abs.shape[0]
                T_pred = pred_abs.shape[0]
                total_ann = T_obs + T_pred

                # Pre-concatenate full timeline for interpolation
                full_world = np.concatenate([obs_abs_np, gt_abs_np], axis=0)  # [total_ann, N, 2]
                if H is not None:
                    full_pix = map_seq(full_world)
                else:
                    full_pix = full_world

                # Video frame start aligned to first annotation of this sequence
                base0 = int(test_dataset.frame_list[0]) if hasattr(test_dataset, 'frame_list') else 0
                abs_frame = int(test_dataset.frame_list[batch_idx]) if hasattr(test_dataset, 'frame_list') else batch_idx
                frame_start = int(args.start_frame + (abs_frame - base0))

                # For each video frame spanning this sequence, interpolate positions
                for k in range(total_ann * step):
                    t_float = k / float(step)  # between 0 and total_ann-1 with fraction
                    t0 = int(np.floor(t_float))
                    t1 = min(t0 + 1, total_ann - 1)
                    alpha = float(t_float - t0)

                    # Linear interpolation between annotation steps
                    interp = (1.0 - alpha) * full_pix[t0] + alpha * full_pix[t1]  # [N,2]

                    # Split back into obs/pred for styling
                    # up to T_obs-1: observed, from T_obs: future
                    obs_k = full_pix[max(0, min(t0, T_obs - 1))]  # use last obs for tail
                    gt_k = interp if t_float >= (T_obs - 1) else full_pix[min(t0, T_obs - 1)]

                    # Draw on the corresponding video frame
                    frame_index = frame_start + k
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        break

                    # Assemble minimal tensors for drawer
                    obs_draw = full_pix[:min(T_obs, t0 + 1)]  # past obs
                    pred_draw = interp[np.newaxis, ...] if t_float >= (T_obs - 1) else np.zeros((0, full_pix.shape[1], 2))
                    gt_draw = full_pix[T_obs: t0 + 1] if t0 + 1 > T_obs else np.zeros((0, full_pix.shape[1], 2))

                    frame_overlay = draw_trajectories_on_frame(frame, obs_draw, pred_draw if pred_draw.size else np.zeros((1, full_pix.shape[1], 2)), gt_draw)
                    writer.write(frame_overlay)
            
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

    # Cleanup video IO
    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()


if __name__ == '__main__':
    main()
