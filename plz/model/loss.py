import torch
import torch.nn as nn
import numpy as np


def generate_statistics_matrices(V):
    """Generate mean and covariance matrices from the network output."""
    mu = V[:, :, 0:2]
    sx = V[:, :, 2].exp()
    sy = V[:, :, 3].exp()
    corr = V[:, :, 4].tanh()

    cov = torch.zeros(V.size(0), V.size(1), 2, 2, device=V.device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy

    return mu, cov


def multivariate_loss(V_pred, V_trgt, training=False):
    """
    Compute multivariate loss for trajectory prediction
    
    Args:
        V_pred: Predicted trajectories with distribution parameters [batch, time, ped, 5]
        V_trgt: Ground truth trajectories [batch, time, ped, 2]
        training: Whether in training mode
    """
    # Extract distribution parameters
    normx = V_trgt[:, :, :, 0] - V_pred[:, :, :, 0]
    normy = V_trgt[:, :, :, 1] - V_pred[:, :, :, 1]

    sx = torch.exp(V_pred[:, :, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, :, 4])  # corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    
    if training:
        # During training, take mean over all dimensions
        result = torch.mean(result)
    else:
        # During validation/testing, preserve batch dimension for metrics
        result = torch.mean(result, dim=[1, 2])
    
    return result


def enhanced_multivariate_loss(V_pred, V_trgt, group_indices=None, velocity=None, 
                               alpha=1.0, beta=0.1, gamma=0.05):
    """
    Enhanced multivariate loss with group and velocity awareness
    
    Args:
        V_pred: Predicted trajectories with distribution parameters
        V_trgt: Ground truth trajectories
        group_indices: Group membership indices
        velocity: Velocity information
        alpha: Weight for basic prediction loss
        beta: Weight for group consistency loss
        gamma: Weight for velocity consistency loss
    """
    # Basic multivariate loss
    basic_loss = multivariate_loss(V_pred, V_trgt, training=True)
    
    total_loss = alpha * basic_loss
    loss_dict = {'basic_loss': basic_loss}
    
    # Group consistency loss
    if group_indices is not None and beta > 0:
        group_loss = compute_group_consistency_loss(V_pred, V_trgt, group_indices)
        total_loss += beta * group_loss
        loss_dict['group_loss'] = group_loss
    
    # Velocity consistency loss
    if velocity is not None and gamma > 0:
        velocity_loss = compute_velocity_consistency_loss(V_pred, velocity)
        total_loss += gamma * velocity_loss
        loss_dict['velocity_loss'] = velocity_loss
    
    loss_dict['total_loss'] = total_loss
    return total_loss, loss_dict


def compute_group_consistency_loss(V_pred, V_trgt, group_indices):
    """
    Compute group consistency loss to encourage coherent group behavior
    """
    if group_indices is None:
        return torch.tensor(0.0, device=V_pred.device)
    
    group_loss = 0.0
    unique_groups = group_indices.unique()
    count = 0
    
    for group_id in unique_groups:
        group_mask = (group_indices == group_id)
        group_size = group_mask.sum()
        
        if group_size > 1:  # Only consider groups with multiple members
            # Get predictions and targets for group members
            group_pred_mean = V_pred[:, :, group_mask, :2].mean(dim=2, keepdim=True)
            group_trgt_mean = V_trgt[:, :, group_mask, :].mean(dim=2, keepdim=True)
            
            # Encourage group members to move coherently
            for i in range(group_size):
                member_pred = V_pred[:, :, group_mask, :2][:, :, i:i+1, :]
                member_trgt = V_trgt[:, :, group_mask, :][:, :, i:i+1, :]
                
                # Loss for deviation from group mean
                pred_deviation = member_pred - group_pred_mean
                trgt_deviation = member_trgt - group_trgt_mean
                group_loss += torch.mean((pred_deviation - trgt_deviation) ** 2)
                count += 1
    
    return group_loss / max(count, 1)


def compute_velocity_consistency_loss(V_pred, velocity):
    """
    Compute velocity consistency loss
    """
    if V_pred.size(1) <= 1:
        return torch.tensor(0.0, device=V_pred.device)
    
    # Extract predicted positions
    pred_positions = V_pred[:, :, :, :2]
    
    # Compute predicted velocities
    pred_velocity = pred_positions[:, 1:, :, :] - pred_positions[:, :-1, :, :]
    
    # Target velocity (should match input velocity pattern)
    if velocity.size(1) > pred_velocity.size(1):
        target_velocity = velocity[:, :pred_velocity.size(1), :, :]
    else:
        target_velocity = velocity
    
    # MSE loss between predicted and target velocities
    velocity_loss = torch.mean((pred_velocity - target_velocity) ** 2)
    
    return velocity_loss


def compute_social_loss(V_pred, adjacency_matrices, collision_threshold=0.2):
    """
    Compute social interaction loss to encourage realistic social behavior
    """
    if V_pred.size(1) <= 1:
        return torch.tensor(0.0, device=V_pred.device)
    
    batch_size, time_steps, num_ped, _ = V_pred.shape
    social_loss = 0.0
    
    pred_positions = V_pred[:, :, :, :2]
    
    for t in range(time_steps):
        positions = pred_positions[:, t, :, :]  # [batch, num_ped, 2]
        
        # Compute pairwise distances
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, num_ped, num_ped, 2]
        distances = torch.norm(pos_diff, p=2, dim=-1)  # [batch, num_ped, num_ped]
        
        # Collision avoidance loss
        collision_mask = (distances < collision_threshold) & (distances > 0)
        if collision_mask.any():
            collision_loss = torch.mean(torch.clamp(collision_threshold - distances[collision_mask], min=0))
            social_loss += collision_loss
        
        # Social force loss (encourage maintaining comfortable distances)
        if adjacency_matrices is not None and t < adjacency_matrices.size(2):
            adj_matrix = adjacency_matrices[:, 0, t, :, :]  # Use first relation
            social_distances = distances * adj_matrix
            comfortable_distance = 1.5
            distance_loss = torch.mean(torch.abs(social_distances - comfortable_distance) * adj_matrix)
            social_loss += 0.1 * distance_loss
    
    return social_loss / time_steps


def compute_trajectory_smoothness_loss(V_pred):
    """
    Compute trajectory smoothness loss to encourage smooth motion
    """
    if V_pred.size(1) <= 2:
        return torch.tensor(0.0, device=V_pred.device)
    
    pred_positions = V_pred[:, :, :, :2]
    
    # First derivative (velocity)
    velocity = pred_positions[:, 1:, :, :] - pred_positions[:, :-1, :, :]
    
    # Second derivative (acceleration)
    acceleration = velocity[:, 1:, :, :] - velocity[:, :-1, :, :]
    
    # Penalize high acceleration (encourage smooth motion)
    smoothness_loss = torch.mean(acceleration ** 2)
    
    return smoothness_loss


def bivariate_loss(V_pred, V_trgt):
    """
    Bivariate loss function for compatibility with GPGraph
    """
    # Extract mean and variance parameters
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx = torch.exp(V_pred[:, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, 4])  # corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result


class ComprehensiveLoss(nn.Module):
    """
    Comprehensive loss function combining all loss components
    """
    def __init__(self, weights=None):
        super().__init__()
        
        # Default weights
        default_weights = {
            'prediction': 1.0,
            'group_consistency': 0.1,
            'velocity_consistency': 0.05,
            'social': 0.02,
            'smoothness': 0.01
        }
        
        self.weights = weights if weights is not None else default_weights
    
    def forward(self, V_pred, V_trgt, group_indices=None, velocity=None, 
                adjacency_matrices=None):
        """
        Compute comprehensive loss
        """
        losses = {}
        total_loss = 0.0
        
        # Prediction loss
        pred_loss = multivariate_loss(V_pred, V_trgt, training=True)
        losses['prediction'] = pred_loss
        total_loss += self.weights['prediction'] * pred_loss
        
        # Group consistency loss
        if group_indices is not None:
            group_loss = compute_group_consistency_loss(V_pred, V_trgt, group_indices)
            losses['group_consistency'] = group_loss
            total_loss += self.weights['group_consistency'] * group_loss
        
        # Velocity consistency loss
        if velocity is not None:
            vel_loss = compute_velocity_consistency_loss(V_pred, velocity)
            losses['velocity_consistency'] = vel_loss
            total_loss += self.weights['velocity_consistency'] * vel_loss
        
        # Social interaction loss
        if adjacency_matrices is not None:
            social_loss = compute_social_loss(V_pred, adjacency_matrices)
            losses['social'] = social_loss
            total_loss += self.weights['social'] * social_loss
        
        # Smoothness loss
        smoothness_loss = compute_trajectory_smoothness_loss(V_pred)
        losses['smoothness'] = smoothness_loss
        total_loss += self.weights['smoothness'] * smoothness_loss
        
        losses['total'] = total_loss
        return total_loss, losses
