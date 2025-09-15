import math
import torch
import numpy as np


def ade(pred_traj, true_traj, masks=None):
    """
    Calculate Average Displacement Error
    
    Args:
        pred_traj: Predicted trajectories [batch, time, ped, 2]
        true_traj: Ground truth trajectories [batch, time, ped, 2]
        masks: Optional masks for valid predictions
    """
    diff = pred_traj - true_traj
    dist = torch.norm(diff, p=2, dim=-1)
    
    if masks is not None:
        dist = dist * masks
        return torch.sum(dist) / torch.sum(masks)
    else:
        return torch.mean(dist)


def fde(pred_traj, true_traj, masks=None):
    """
    Calculate Final Displacement Error
    
    Args:
        pred_traj: Predicted trajectories [batch, time, ped, 2]
        true_traj: Ground truth trajectories [batch, time, ped, 2]
        masks: Optional masks for valid predictions
    """
    diff = pred_traj[:, -1] - true_traj[:, -1]  # Final positions
    dist = torch.norm(diff, p=2, dim=-1)
    
    if masks is not None:
        final_masks = masks[:, -1]
        dist = dist * final_masks
        return torch.sum(dist) / torch.sum(final_masks)
    else:
        return torch.mean(dist)


def compute_batch_metric(pred, gt):
    """
    Compute ADE, FDE, COL, TCC scores for each pedestrian
    
    Args:
        pred: Predicted trajectories [n_samples, time, ped, 2]
        gt: Ground truth trajectories [time, ped, 2]
    
    Returns:
        ADEs, FDEs, COLs, TCCs for each pedestrian
    """
    # Calculate distances
    temp = (pred - gt).norm(p=2, dim=-1)  # [n_samples, time, ped]
    
    # ADE: Average over time, minimum over samples
    ADEs = temp.mean(dim=1).min(dim=0)[0]  # [ped]
    
    # FDE: Final displacement, minimum over samples
    FDEs = temp[:, -1, :].min(dim=0)[0]  # [ped]
    
    # Get best predictions
    best_sample_idx = temp[:, -1, :].argmin(dim=0)  # [ped]
    pred_best = pred[best_sample_idx, :, range(pred.size(2)), :]  # [time, ped, 2]
    
    # TCC: Trajectory Correlation Coefficient
    pred_gt_stack = torch.stack([pred_best, gt], dim=0)  # [2, time, ped, 2]
    pred_gt_stack = pred_gt_stack.permute(2, 1, 0, 3)  # [ped, time, 2, 2]
    
    # Calculate covariance
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    
    # Calculate correlation coefficient
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)  # [ped]
    
    # COL: Collision rate
    num_interp, thres = 4, 0.2
    pred_fp = pred[:, [0], :, :]  # First position
    pred_rel = pred[:, 1:] - pred[:, :-1]  # Relative movements
    
    # Interpolate for denser collision checking
    pred_rel_dense = pred_rel.div(num_interp).unsqueeze(dim=2).repeat_interleave(
        repeats=num_interp, dim=2).contiguous()
    pred_rel_dense = pred_rel_dense.reshape(
        pred.size(0), num_interp * (pred.size(1) - 1), pred.size(2), pred.size(3))
    pred_dense = torch.cat([pred_fp, pred_rel_dense], dim=1).cumsum(dim=1)
    
    # Check collisions
    col_mask = pred_dense[:, :3 * num_interp + 2].unsqueeze(dim=2).repeat_interleave(
        repeats=pred.size(2), dim=2)
    col_mask = (col_mask - col_mask.transpose(2, 3)).norm(p=2, dim=-1)
    col_mask = col_mask.add(torch.eye(n=pred.size(2), device=pred.device)[None, None, :, :])
    col_mask = col_mask.min(dim=1)[0].lt(thres)
    COLs = col_mask.sum(dim=1).gt(0).type(pred.type()).mean(dim=0).mul(100)
    
    return ADEs, FDEs, COLs, TCCs


def compute_metrics(pred_samples, gt_traj, obs_traj=None):
    """
    Compute comprehensive metrics for trajectory prediction
    
    Args:
        pred_samples: Predicted trajectory samples [n_samples, batch, time, ped, 2]
        gt_traj: Ground truth trajectories [batch, time, ped, 2]
        obs_traj: Observed trajectories [batch, time, ped, 2] (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Convert to absolute coordinates if needed
    if obs_traj is not None:
        # Assume pred_samples are relative, convert to absolute
        last_obs = obs_traj[:, -1:, :, :]  # [batch, 1, ped, 2]
        pred_abs = torch.cumsum(pred_samples, dim=2) + last_obs.unsqueeze(0)
    else:
        pred_abs = pred_samples
    
    batch_size = gt_traj.size(0)
    ade_all = []
    fde_all = []
    col_all = []
    tcc_all = []
    
    for batch_idx in range(batch_size):
        pred_batch = pred_abs[:, batch_idx]  # [n_samples, time, ped, 2]
        gt_batch = gt_traj[batch_idx]  # [time, ped, 2]
        
        # Compute metrics for this batch
        ADEs, FDEs, COLs, TCCs = compute_batch_metric(pred_batch, gt_batch)
        
        ade_all.append(ADEs.detach().cpu().numpy())
        fde_all.append(FDEs.detach().cpu().numpy())
        col_all.append(COLs.detach().cpu().numpy())
        tcc_all.append(TCCs.detach().cpu().numpy())
    
    # Aggregate metrics
    ade_all = np.concatenate(ade_all)
    fde_all = np.concatenate(fde_all)
    col_all = np.concatenate(col_all)
    tcc_all = np.concatenate(tcc_all)
    
    metrics['ADE'] = np.mean(ade_all)
    metrics['FDE'] = np.mean(fde_all)
    metrics['COL'] = np.mean(col_all)
    metrics['TCC'] = np.mean(tcc_all)
    
    # Additional statistics
    metrics['ADE_std'] = np.std(ade_all)
    metrics['FDE_std'] = np.std(fde_all)
    metrics['ADE_min'] = np.min(ade_all)
    metrics['FDE_min'] = np.min(fde_all)
    metrics['ADE_max'] = np.max(ade_all)
    metrics['FDE_max'] = np.max(fde_all)
    
    return metrics


def compute_group_metrics(pred_samples, gt_traj, group_indices):
    """
    Compute group-specific metrics
    
    Args:
        pred_samples: Predicted trajectory samples
        gt_traj: Ground truth trajectories
        group_indices: Group membership indices
    
    Returns:
        Dictionary of group metrics
    """
    metrics = {}
    unique_groups = group_indices.unique()
    
    group_ades = []
    group_fdes = []
    group_coherence = []
    
    for group_id in unique_groups:
        group_mask = (group_indices == group_id)
        group_size = group_mask.sum().item()
        
        if group_size > 1:
            # Extract group predictions and ground truth
            group_pred = pred_samples[:, :, :, group_mask, :]
            group_gt = gt_traj[:, :, group_mask, :]
            
            # Compute group ADE/FDE
            group_metrics = compute_metrics(group_pred, group_gt)
            group_ades.append(group_metrics['ADE'])
            group_fdes.append(group_metrics['FDE'])
            
            # Compute group coherence (how well group members move together)
            coherence = compute_group_coherence(group_pred, group_gt)
            group_coherence.append(coherence)
    
    if group_ades:
        metrics['group_ADE'] = np.mean(group_ades)
        metrics['group_FDE'] = np.mean(group_fdes)
        metrics['group_coherence'] = np.mean(group_coherence)
    else:
        metrics['group_ADE'] = 0.0
        metrics['group_FDE'] = 0.0
        metrics['group_coherence'] = 0.0
    
    return metrics


def compute_group_coherence(group_pred, group_gt):
    """
    Compute how coherently group members move together
    
    Args:
        group_pred: Predicted trajectories for group members
        group_gt: Ground truth trajectories for group members
    
    Returns:
        Group coherence score
    """
    # Calculate center of mass for predictions and ground truth
    pred_center = group_pred.mean(dim=-2, keepdim=True)  # [samples, batch, time, 1, 2]
    gt_center = group_gt.mean(dim=-2, keepdim=True)  # [batch, time, 1, 2]
    
    # Calculate deviations from center
    pred_deviations = group_pred - pred_center
    gt_deviations = group_gt.unsqueeze(0) - gt_center.unsqueeze(0)
    
    # Compute coherence as similarity of deviation patterns
    coherence = torch.mean((pred_deviations - gt_deviations) ** 2)
    
    return coherence.item()


def compute_velocity_metrics(pred_samples, gt_traj):
    """
    Compute velocity-related metrics
    """
    # Calculate predicted velocities
    pred_vel = pred_samples[:, :, 1:] - pred_samples[:, :, :-1]
    gt_vel = gt_traj[:, 1:] - gt_traj[:, :-1]
    
    # Velocity ADE
    vel_diff = pred_vel - gt_vel.unsqueeze(0)
    vel_ade = torch.mean(torch.norm(vel_diff, p=2, dim=-1))
    
    # Speed accuracy
    pred_speed = torch.norm(pred_vel, p=2, dim=-1)
    gt_speed = torch.norm(gt_vel, p=2, dim=-1)
    speed_error = torch.mean(torch.abs(pred_speed - gt_speed.unsqueeze(0)))
    
    return {
        'velocity_ADE': vel_ade.item(),
        'speed_error': speed_error.item()
    }
