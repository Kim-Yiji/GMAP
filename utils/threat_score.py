"""
Threat Score Calculation Module for DMRGCN

This module calculates threat scores T_{ij} and feature vectors z_{ij} for each edge (i, j)
in the graph, where i is a pedestrian and j is an obstacle (pedestrian or other object).

The threat score is computed based on:
1. Distance d_{ij}
2. Approach velocity v^{+}_{ij}
3. Obstacle size size_j
4. Time-to-Collision TTC_{ij}

All calculations are performed in frame units (not seconds).
"""

import torch
import torch.nn as nn
import numpy as np


def get_obstacle_size(obstacle_type=None, default_size=0.0):
    """
    Get obstacle size value based on obstacle type.
    
    Args:
        obstacle_type: Type of obstacle (string or int identifier)
        default_size: Default size if type is unknown (default: 0.0 for human size)
    
    Returns:
        size_j: Scalar size value
            - 0.0: Human size (pedestrian, skateboarder, person pushing stroller)
            - 0.2: Similar to human size (bicycle, shopping cart)
            - 0.7: Larger than human (car)
            - 1.0: Much larger than human (truck, bus, train)
    """
    if obstacle_type is None:
        return default_size
    
    # Map obstacle types to size values
    size_mapping = {
        # Human size (0.0)
        'pedestrian': 0.0,
        'person': 0.0,
        'skateboarder': 0.0,
        'skater': 0.0,  # SDD dataset
        'stroller': 0.0,  # Person pushing stroller (human-sized)
        # Similar to human size (0.2)
        'bicycle': 0.2,
        'bike': 0.2,
        'biker': 0.2,  # SDD dataset - bicycle rider (similar to human)
        'shopping_cart': 0.2,
        'cart': 0.2,  # SDD dataset
        # Larger than human (0.7)
        'car': 0.7,  # SDD dataset
        'vehicle': 0.7,
        # Much larger than human (1.0)
        'truck': 1.0,
        'bus': 1.0,  # SDD dataset
        'train': 1.0,
    }
    
    # Handle string input
    if isinstance(obstacle_type, str):
        obstacle_type_lower = obstacle_type.lower()
        return size_mapping.get(obstacle_type_lower, default_size)
    
    # Handle integer/other types - assume default (human size)
    return default_size


def compute_relative_vectors(obs_traj, obs_traj_rel, pedestrian_mask=None):
    """
    Compute relative position and velocity vectors for pairs (i, j).
    
    Only computes for pedestrians (i) and all objects (j).
    i: pedestrians only
    j: all objects (pedestrians and non-pedestrians)
    
    Uses vectorized operations for efficiency.
    
    Args:
        obs_traj: Absolute positions, shape (num_peds, 2, seq_len)
        obs_traj_rel: Relative velocities, shape (num_peds, 2, seq_len)
        pedestrian_mask: Boolean mask indicating which objects are pedestrians, shape (num_peds,)
            If None, assumes all objects are pedestrians (backward compatibility)
    
    Returns:
        rel_pos: Relative position vectors r_{ij} = p_j - p_i, shape (num_pedestrians, num_peds, 2, seq_len)
            Only computed for pedestrians (i)
        rel_vel: Relative velocity vectors v_{ij} = v_j - v_i, shape (num_pedestrians, num_peds, 2, seq_len)
            Only computed for pedestrians (i)
    """
    num_peds = obs_traj.shape[0]
    device = obs_traj.device
    
    # If no pedestrian mask, compute for all objects
    if pedestrian_mask is None:
        pedestrian_indices = list(range(num_peds))
    else:
        if isinstance(pedestrian_mask, (list, np.ndarray)):
            pedestrian_mask = torch.tensor(pedestrian_mask, dtype=torch.bool, device=device)
        pedestrian_indices = torch.where(pedestrian_mask)[0].tolist()
    
    num_pedestrians = len(pedestrian_indices)
    
    if num_pedestrians == 0:
        # No pedestrians, return empty tensors
        return torch.zeros((0, num_peds, 2, obs_traj.shape[2]), dtype=obs_traj.dtype, device=device), \
               torch.zeros((0, num_peds, 2, obs_traj_rel.shape[2]), dtype=obs_traj_rel.dtype, device=device)
    
    # Extract pedestrian trajectories (i)
    obs_traj_pedestrians = obs_traj[pedestrian_indices, :, :]  # (num_pedestrians, 2, seq_len)
    obs_traj_rel_pedestrians = obs_traj_rel[pedestrian_indices, :, :]  # (num_pedestrians, 2, seq_len)
    
    # Vectorized computation: expand dimensions for broadcasting
    # obs_traj_pedestrians: (num_pedestrians, 2, seq_len) -> (num_pedestrians, 1, 2, seq_len)
    # obs_traj: (num_peds, 2, seq_len) -> (1, num_peds, 2, seq_len)
    obs_traj_i = obs_traj_pedestrians.unsqueeze(1)  # (num_pedestrians, 1, 2, seq_len)
    obs_traj_j = obs_traj.unsqueeze(0)  # (1, num_peds, 2, seq_len)
    
    obs_traj_rel_i = obs_traj_rel_pedestrians.unsqueeze(1)  # (num_pedestrians, 1, 2, seq_len)
    obs_traj_rel_j = obs_traj_rel.unsqueeze(0)  # (1, num_peds, 2, seq_len)
    
    # r_{ij} = p_j - p_i (only for pedestrians i)
    rel_pos = obs_traj_j - obs_traj_i  # (num_pedestrians, num_peds, 2, seq_len)
    
    # v_{ij} = v_j - v_i (only for pedestrians i)
    rel_vel = obs_traj_rel_j - obs_traj_rel_i  # (num_pedestrians, num_peds, 2, seq_len)
    
    return rel_pos, rel_vel


def compute_threat_features(rel_pos, rel_vel, obstacle_sizes=None, eps=1e-6, ttc_max=20.0):
    """
    Compute threat feature vector z_{ij} for each edge (i, j).
    
    i: pedestrians only (from rel_pos/rel_vel shape)
    j: all objects (pedestrians and non-pedestrians)
    
    Uses vectorized operations for efficiency.
    
    Args:
        rel_pos: Relative position vectors r_{ij}, shape (num_pedestrians, num_peds, 2, seq_len)
            Only computed for pedestrians (i)
        rel_vel: Relative velocity vectors v_{ij}, shape (num_pedestrians, num_peds, 2, seq_len)
            Only computed for pedestrians (i)
        obstacle_sizes: Size values for each obstacle j, shape (num_peds,) or None
        eps: Small epsilon for numerical stability (default: 1e-6)
        ttc_max: Maximum TTC value for capping (default: 20.0 frames)
    
    Returns:
        z_ij: Feature vectors, shape (num_pedestrians, num_peds, 4, seq_len)
            z_ij[i, j, :, t] = [d_{ij}[t], v^{+}_{ij}[t], size_j, TTC_{ij}[t]]
            Only computed for pedestrians (i)
    """
    num_pedestrians = rel_pos.shape[0]
    num_peds = rel_pos.shape[1]
    seq_len = rel_pos.shape[3]
    device = rel_pos.device
    dtype = rel_pos.dtype
    
    # Default obstacle sizes (assume all are human-sized if not provided)
    if obstacle_sizes is None:
        obstacle_sizes = torch.zeros(num_peds, dtype=dtype, device=device)
    elif isinstance(obstacle_sizes, (list, np.ndarray)):
        obstacle_sizes = torch.tensor(obstacle_sizes, dtype=dtype, device=device)
    elif not isinstance(obstacle_sizes, torch.Tensor):
        obstacle_sizes = torch.zeros(num_peds, dtype=dtype, device=device)
    
    # Ensure obstacle_sizes has correct shape and expand for broadcasting
    if obstacle_sizes.shape[0] != num_peds:
        obstacle_sizes = torch.zeros(num_peds, dtype=dtype, device=device)
    
    # Expand obstacle_sizes: (num_peds,) -> (num_pedestrians, num_peds, 1, seq_len)
    obstacle_sizes_expanded = obstacle_sizes.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(num_pedestrians, num_peds, 1, seq_len)
    
    # 1. Distance d_{ij} = ||r_{ij}||
    # rel_pos: (num_pedestrians, num_peds, 2, seq_len)
    d_ij = torch.norm(rel_pos, p=2, dim=2)  # (num_pedestrians, num_peds, seq_len)
    d_ij_safe = d_ij + eps  # Avoid division by zero
    
    # 2. Approach velocity v^{+}_{ij} = max(0, -r_{ij} · v_{ij} / ||r_{ij}||)
    # Dot product: sum over dimension 2 (spatial dimension)
    dot_product = torch.sum(rel_pos * rel_vel, dim=2)  # (num_pedestrians, num_peds, seq_len)
    v_plus_ij = torch.clamp(-dot_product / d_ij_safe, min=0.0)  # (num_pedestrians, num_peds, seq_len)
    
    # 3. Obstacle size size_j (already expanded above)
    size_j = obstacle_sizes_expanded.squeeze(2)  # (num_pedestrians, num_peds, seq_len)
    
    # 4. Time-to-Collision TTC_{ij} = d_{ij} / max(eps, v^{+}_{ij})
    v_plus_safe = v_plus_ij + eps
    ttc_ij = d_ij / v_plus_safe
    
    # Cap TTC at maximum value when v^{+}_{ij} is zero or very small
    ttc_ij = torch.where(v_plus_ij < eps, 
                        torch.full_like(ttc_ij, ttc_max),
                        torch.clamp(ttc_ij, min=0.0, max=ttc_max))
    
    # Stack features: [d, v+, size, TTC]
    z_ij = torch.stack([d_ij, v_plus_ij, size_j, ttc_ij], dim=2)  # (num_pedestrians, num_peds, 4, seq_len)
    
    # Set self-connections (i == j where both are pedestrians) to zero
    # Note: i is always a pedestrian, but j might not be, so we only zero when j is also a pedestrian
    # This is handled later in the pipeline if needed
    
    return z_ij


def normalize_threat_features_minmax(z_ij, reverse_direction=None):
    """
    Apply min-max normalization to threat features with direction reversal for threat-increasing variables.
    
    Step 1: Min-Max Normalization
    - Normalize each variable to [0, 1] range using dataset-level min/max
    - Reverse direction for variables where smaller values mean greater threat (d, TTC)
    
    Args:
        z_ij: Feature vectors, shape (num_pedestrians, num_peds, 4, seq_len)
            z_ij[:, :, 0, :] = d_ij (distance)
            z_ij[:, :, 1, :] = v+_ij (approach velocity)
            z_ij[:, :, 2, :] = size_j (obstacle size)
            z_ij[:, :, 3, :] = TTC_ij (time-to-collision)
            Only computed for pedestrians (i)
        reverse_direction: List of feature indices to reverse (default: [0, 3] for d and TTC)
    
    Returns:
        z_ij_norm: Normalized feature vectors, same shape as z_ij
            All values in [0, 1], with higher values meaning greater threat
    """
    if reverse_direction is None:
        reverse_direction = [0, 3]  # d_ij and TTC_ij (smaller = more threat)
    
    num_pedestrians = z_ij.shape[0]
    num_peds = z_ij.shape[1]
    seq_len = z_ij.shape[3]
    device = z_ij.device
    
    z_ij_norm = z_ij.clone()
    
    # Normalize each feature dimension separately
    for feat_idx in range(4):  # [d, v+, size, TTC]
        feat_values = z_ij[:, :, feat_idx, :]  # (num_pedestrians, num_peds, seq_len)
        
        # Get all values (no need to exclude self-connections since i is always pedestrian, j might not be)
        feat_values_flat = feat_values.flatten()
        
        if len(feat_values_flat) == 0:
            continue
        
        # Min-max normalization: x' = (x - x_min) / (x_max - x_min)
        feat_min = feat_values_flat.min()
        feat_max = feat_values_flat.max()
        
        if feat_max > feat_min:
            # Normalize to [0, 1]
            feat_norm = (feat_values - feat_min) / (feat_max - feat_min)
            
            # Reverse direction for threat-increasing variables (d, TTC)
            # d'_ij = 1 - (d_ij - d_min) / (d_max - d_min)
            # TTC'_ij = 1 - (TTC_ij - TTC_min) / (TTC_max - TTC_min)
            if feat_idx in reverse_direction:
                feat_norm = 1.0 - feat_norm
            
            z_ij_norm[:, :, feat_idx, :] = feat_norm
        else:
            # All values are the same, set to 0.5 (neutral)
            z_ij_norm[:, :, feat_idx, :] = torch.full_like(feat_values, 0.5)
    
    return z_ij_norm


def compute_threat_score(z_ij, weights=None, tau=0.15, beta=0.5, num_total_objects=None, pedestrian_indices=None):
    """
    Compute threat score T_{ij} from feature vector z_{ij} using the specified pipeline.
    
    z_ij is computed only for pedestrians (i) and all objects (j).
    This function expands the result to full size (num_total_objects, num_total_objects, seq_len)
    where non-pedestrians (i) have threat_score[i, :, :] = 0.
    
    Step 1: Min-Max Normalization with direction reversal
    Step 2: Linear weighted combination
    Step 3: Sigmoid transformation with temperature and bias
    
    Args:
        z_ij: Feature vectors, shape (num_pedestrians, num_peds, 4, seq_len)
            z_ij[:, :, 0, :] = d_ij (distance)
            z_ij[:, :, 1, :] = v+_ij (approach velocity)
            z_ij[:, :, 2, :] = size_j (obstacle size)
            z_ij[:, :, 3, :] = TTC_ij (time-to-collision)
            Only computed for pedestrians (i)
        weights: Weights for each feature [w_d, w_v, w_size, w_ttc], shape (4,) or None
            If None, uses distance-weighted [0.5, 0.25, 0.15, 0.1] (distance has the largest impact)
        tau: Temperature parameter for sigmoid (controls slope), default=0.15
        beta: Midpoint parameter for sigmoid (controls center), default=0.5
        num_total_objects: Total number of objects (including non-pedestrians), for output shape
        pedestrian_indices: List of indices of pedestrians in the full object list
    
    Returns:
        threat_score: Threat scores T_{ij}, shape (num_total_objects, num_total_objects, seq_len)
            Values are in [0, 1] range
            threat_score[i, j, t] = threat perceived by pedestrian i from obstacle j at frame t
            If i is not a pedestrian, threat_score[i, :, :] = 0 (not computed)
    """
    num_pedestrians = z_ij.shape[0]
    num_peds = z_ij.shape[1]
    seq_len = z_ij.shape[3]
    device = z_ij.device
    
    if num_total_objects is None:
        num_total_objects = num_peds
    
    # Initialize threat_score with zeros (non-pedestrians will remain 0)
    threat_score = torch.zeros((num_total_objects, num_total_objects, seq_len), dtype=z_ij.dtype, device=device)
    
    if num_pedestrians == 0:
        # No pedestrians, return all zeros
        return threat_score
    
    # Step 1: Min-Max Normalization with direction reversal
    z_ij_norm = normalize_threat_features_minmax(z_ij, reverse_direction=[0, 3])
    
    # Step 2: Linear weighted combination
    # Default weights: distance-weighted (distance has the largest impact based on empirical observations)
    # [w_d, w_v, w_size, w_ttc] = [distance, approach_velocity, obstacle_size, TTC]
    if weights is None:
        weights = torch.tensor([0.5, 0.25, 0.15, 0.1], dtype=z_ij.dtype, device=device)
    elif isinstance(weights, (list, np.ndarray)):
        weights = torch.tensor(weights, dtype=z_ij.dtype, device=device)
    
    # Ensure weights have correct shape for broadcasting: (4,) -> (1, 1, 4, 1)
    weights = weights.view(1, 1, 4, 1)
    
    # Compute weighted sum: u_{ij} = w_d * d'_ij + w_v * v'^+_ij + w_s * size'_j + w_T * TTC'_ij
    # z_ij_norm: (num_pedestrians, num_peds, 4, seq_len)
    # weights: (1, 1, 4, 1)
    u_ij = torch.sum(weights * z_ij_norm, dim=2)  # (num_pedestrians, num_peds, seq_len)
    
    # Step 3: Apply sigmoid transformation
    # Threat_{ij} = 1 / (1 + exp(-(u_ij - beta) / tau))
    # This ensures output values lie in [0, 1] and vary smoothly
    threat_score_pedestrians = torch.sigmoid((u_ij - beta) / tau)
    
    # Set self-connections (i == j where both are pedestrians) to zero
    # Note: i is always a pedestrian, j might be pedestrian or not
    if pedestrian_indices is not None:
        # Create mask for pedestrian-pedestrian pairs (i == j)
        for ped_idx, i in enumerate(pedestrian_indices):
            if i < num_peds:  # j index in the original object list
                threat_score_pedestrians[ped_idx, i, :] = 0.0
    
    # Expand threat_score_pedestrians to full size
    # threat_score_pedestrians: (num_pedestrians, num_peds, seq_len)
    # threat_score: (num_total_objects, num_total_objects, seq_len)
    if pedestrian_indices is not None:
        for ped_idx, i in enumerate(pedestrian_indices):
            threat_score[i, :, :] = threat_score_pedestrians[ped_idx, :, :]
    else:
        # If pedestrian_indices not provided, assume first num_pedestrians are pedestrians
        threat_score[:num_pedestrians, :, :] = threat_score_pedestrians
    
    return threat_score


def compute_threat_score_batch(obs_traj, obs_traj_rel, obstacle_sizes=None, 
                               weights=None, tau=0.15, beta=0.5, eps=1e-6, ttc_max=20.0,
                               pedestrian_mask=None, object_labels=None):
    """
    Complete pipeline to compute threat scores from trajectory data.
    
    Only computes threat scores for pedestrians (i) perceiving threats from obstacles (j).
    Does not compute threats perceived by non-pedestrian objects (e.g., car, bus).
    
    Args:
        obs_traj: Absolute positions, shape (num_peds, 2, seq_len)
        obs_traj_rel: Relative velocities, shape (num_peds, 2, seq_len)
        obstacle_sizes: Size values for each obstacle, shape (num_peds,) or None
        weights: Weights for threat score computation [w_d, w_v, w_size, w_ttc], shape (4,) or None
            If None, uses distance-weighted [0.5, 0.25, 0.15, 0.1] (distance has the largest impact)
        tau: Temperature parameter for sigmoid (controls slope), default=0.15
        beta: Midpoint parameter for sigmoid (controls center), default=0.5
        eps: Small epsilon for numerical stability
        ttc_max: Maximum TTC value for capping
        pedestrian_mask: Boolean mask indicating which objects are pedestrians, shape (num_peds,)
            If None and object_labels is provided, will be inferred from labels
        object_labels: List of object type labels (e.g., ['Pedestrian', 'Biker', 'Car', 'Bus'])
            Used to infer pedestrian_mask if pedestrian_mask is None
            If both are None, a warning is issued and all objects are treated as pedestrians
    
    Returns:
        threat_score: Threat scores T_{ij}, shape (num_peds, num_peds, seq_len)
            Values in [0, 1] range
            threat_score[i, j, t] = threat perceived by pedestrian i from obstacle j at frame t
            If i is not a pedestrian, threat_score[i, :, :] = 0
        z_ij: Feature vectors z_{ij}, shape (num_peds, num_peds, 4, seq_len)
            Raw features before normalization: [d_ij, v+_ij, size_j, TTC_ij]
    """
    num_peds = obs_traj.shape[0]
    
    # Infer pedestrian_mask from object_labels if not provided
    if pedestrian_mask is None:
        if object_labels is not None:
            # Only 'pedestrian' and 'person' are considered pedestrians
            # Biker, Skater, Car, Bus, etc. are NOT pedestrians
            pedestrian_types = {'pedestrian', 'person'}
            pedestrian_mask = [label.lower() in pedestrian_types for label in object_labels]
            pedestrian_mask = torch.tensor(pedestrian_mask, dtype=torch.bool)
            
            # Verify that we have at least some pedestrians
            num_pedestrians = pedestrian_mask.sum().item()
            if num_pedestrians == 0:
                import warnings
                warnings.warn("No pedestrians found in object_labels. All objects will be treated as pedestrians.")
                pedestrian_mask = None
            else:
                num_non_pedestrians = num_peds - num_pedestrians
                if num_non_pedestrians > 0:
                    print(f"  Identified {num_pedestrians} pedestrians and {num_non_pedestrians} non-pedestrians")
        else:
            import warnings
            warnings.warn(
                "Neither pedestrian_mask nor object_labels provided. "
                "All objects will be treated as pedestrians. "
                "To compute threats only for pedestrians, provide object_labels parameter.",
                UserWarning
            )
            pedestrian_mask = None
    
    # Get pedestrian mask and indices
    if pedestrian_mask is None:
        if object_labels is not None:
            pedestrian_types = {'pedestrian', 'person'}
            pedestrian_mask = [label.lower() in pedestrian_types for label in object_labels]
            pedestrian_mask = torch.tensor(pedestrian_mask, dtype=torch.bool)
        else:
            pedestrian_mask = None
    
    if pedestrian_mask is not None:
        if isinstance(pedestrian_mask, (list, np.ndarray)):
            pedestrian_mask = torch.tensor(pedestrian_mask, dtype=torch.bool)
        pedestrian_indices = torch.where(pedestrian_mask)[0].tolist()
    else:
        pedestrian_indices = None
    
    # Step 1: Compute relative vectors (only for pedestrians i)
    rel_pos, rel_vel = compute_relative_vectors(obs_traj, obs_traj_rel, pedestrian_mask)
    
    # Step 2: Compute threat features (only for pedestrians i, all objects j)
    z_ij = compute_threat_features(rel_pos, rel_vel, obstacle_sizes, eps, ttc_max)
    
    # Step 3: Compute threat scores using new pipeline
    num_total_objects = obs_traj.shape[0]
    threat_score = compute_threat_score(z_ij, weights, tau, beta, num_total_objects, pedestrian_indices)
    
    # Expand z_ij to full size for backward compatibility
    # z_ij: (num_pedestrians, num_peds, 4, seq_len) -> (num_peds, num_peds, 4, seq_len)
    if z_ij.shape[0] < num_total_objects:
        z_ij_full = torch.zeros((num_total_objects, num_total_objects, 4, z_ij.shape[3]), 
                                dtype=z_ij.dtype, device=z_ij.device)
        if pedestrian_indices is not None:
            for ped_idx, i in enumerate(pedestrian_indices):
                z_ij_full[i, :, :, :] = z_ij[ped_idx, :, :, :]
        else:
            z_ij_full[:z_ij.shape[0], :, :, :] = z_ij
        z_ij = z_ij_full
    
    return threat_score, z_ij

