# Graph building utilities for DMRGCN + GP-Graph integration
# Handles group-aware graph construction, pooling, and unpooling operations

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


def build_relations(pairwise_dist: torch.Tensor, 
                   rel_disp: torch.Tensor, 
                   scales: List[float] = [0.5, 1.0, 2.0]) -> torch.Tensor:
    """Build multi-relational adjacency matrices based on distance scales
    
    Args:
        pairwise_dist: (B, T, N, N) - pairwise distance matrices
        rel_disp: (B, T, N, N) - relative displacement matrices  
        scales: List of distance scales for relations
        
    Returns:
        A: (B, R, T, N, N) - multi-relational adjacency matrices
    """
    B, T, N, _ = pairwise_dist.shape
    R = len(scales)
    
    # Initialize adjacency tensor
    A = torch.zeros(B, R, T, N, N, device=pairwise_dist.device, dtype=pairwise_dist.dtype)
    
    for r, scale in enumerate(scales):
        # Distance-based adjacency
        dist_adj = (pairwise_dist <= scale).float()
        
        # Combine with displacement information
        # Normalize displacement for weighting
        disp_weights = 1.0 / (rel_disp + 1e-6)  # Avoid division by zero
        disp_weights = torch.where(torch.isinf(disp_weights), 
                                  torch.zeros_like(disp_weights), 
                                  disp_weights)
        
        # Weighted adjacency
        A[:, r, :, :, :] = dist_adj * disp_weights
    
    return A


def mask_by_group(adjacency: torch.Tensor, group_indices: torch.Tensor) -> torch.Tensor:
    """Mask adjacency matrix to only include intra-group connections
    
    Args:
        adjacency: (B, R, T, N, N) - adjacency matrices
        group_indices: (B, N) - group assignment for each agent
        
    Returns:
        masked_adj: (B, R, T, N, N) - masked adjacency with only intra-group edges
    """
    B, R, T, N, _ = adjacency.shape
    
    # Create group mask: same group = 1, different group = 0
    group_mask = torch.zeros(B, N, N, device=adjacency.device, dtype=adjacency.dtype)
    
    for b in range(B):
        for i in range(N):
            for j in range(N):
                if group_indices[b, i] == group_indices[b, j]:
                    group_mask[b, i, j] = 1.0
    
    # Expand mask to match adjacency dimensions and apply
    group_mask = group_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N, N)
    masked_adj = adjacency * group_mask
    
    return masked_adj


def pool_groups(features: torch.Tensor, 
               adjacency: torch.Tensor,
               group_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pool agent-level features and adjacency to group-level
    
    Args:
        features: (B, T, N, F) - agent-level features
        adjacency: (B, R, T, N, N) - agent-level adjacency
        group_indices: (B, N) - group assignment for each agent
        
    Returns:
        group_features: (B, T, Ng, F) - group-level features
        group_adjacency: (B, R, T, Ng, Ng) - group-level adjacency
    """
    B, T, N, F = features.shape
    _, R, _, _, _ = adjacency.shape
    
    # Get unique groups and create mapping
    device = features.device
    max_groups = group_indices.max().item() + 1
    
    # Initialize group-level tensors
    group_features = torch.zeros(B, T, max_groups, F, device=device, dtype=features.dtype)
    group_adjacency = torch.zeros(B, R, T, max_groups, max_groups, device=device, dtype=adjacency.dtype)
    group_counts = torch.zeros(B, max_groups, device=device, dtype=torch.float)
    
    # Pool features by averaging within groups
    for b in range(B):
        for g in range(max_groups):
            group_mask = (group_indices[b] == g)
            if group_mask.sum() > 0:
                group_features[b, :, g, :] = features[b, :, group_mask, :].mean(dim=1)
                group_counts[b, g] = group_mask.sum().float()
    
    # Pool adjacency by creating inter-group connections
    for b in range(B):
        for r in range(R):
            for t in range(T):
                for i in range(N):
                    for j in range(N):
                        gi = group_indices[b, i].item()
                        gj = group_indices[b, j].item()
                        if gi != gj and adjacency[b, r, t, i, j] > 0:
                            # Connect different groups if there's an agent-level connection
                            group_adjacency[b, r, t, gi, gj] = 1.0
                            group_adjacency[b, r, t, gj, gi] = 1.0
    
    return group_features, group_adjacency


def unpool_groups(group_features: torch.Tensor, group_indices: torch.Tensor) -> torch.Tensor:
    """Unpool group-level features back to agent-level
    
    Args:
        group_features: (B, T, Ng, F) - group-level features
        group_indices: (B, N) - group assignment for each agent
        
    Returns:
        agent_features: (B, T, N, F) - unpooled agent-level features
    """
    B, T, Ng, F = group_features.shape
    N = group_indices.shape[1]
    
    # Initialize agent-level features
    agent_features = torch.zeros(B, T, N, F, device=group_features.device, dtype=group_features.dtype)
    
    # Assign group features to agents
    for b in range(B):
        for n in range(N):
            g = group_indices[b, n].item()
            agent_features[b, :, n, :] = group_features[b, :, g, :]
    
    return agent_features


def assign_groups_by_distance(positions: torch.Tensor, 
                            threshold: float = 2.0, 
                            method: str = 'threshold') -> torch.Tensor:
    """Assign agents to groups based on spatial proximity
    
    Args:
        positions: (B, T, N, 2) - agent positions (use last timestep)
        threshold: distance threshold for grouping
        method: 'threshold' or 'kmeans'
        
    Returns:
        group_indices: (B, N) - group assignment for each agent
    """
    B, T, N, _ = positions.shape
    device = positions.device
    
    # Use last timestep for grouping
    last_pos = positions[:, -1, :, :]  # (B, N, 2)
    
    group_indices = torch.zeros(B, N, device=device, dtype=torch.long)
    
    for b in range(B):
        pos = last_pos[b]  # (N, 2)
        
        if method == 'threshold':
            # Threshold-based grouping
            groups_assigned = torch.zeros(N, device=device, dtype=torch.bool)
            current_group = 0
            
            for i in range(N):
                if not groups_assigned[i]:
                    # Start new group
                    group_indices[b, i] = current_group
                    groups_assigned[i] = True
                    
                    # Find nearby agents
                    for j in range(i + 1, N):
                        if not groups_assigned[j]:
                            dist = torch.norm(pos[i] - pos[j], p=2)
                            if dist <= threshold:
                                group_indices[b, j] = current_group
                                groups_assigned[j] = True
                    
                    current_group += 1
        
        elif method == 'kmeans':
            # Simple k-means-like grouping (fixed number of groups)
            # For simplicity, use threshold to estimate number of groups
            distances = torch.pdist(pos)
            close_pairs = (distances <= threshold).sum().item()
            num_groups = max(1, N - close_pairs // 2)
            
            # Random initialization and simple assignment
            centroids = pos[torch.randperm(N)[:num_groups]]
            
            for i in range(N):
                dists_to_centroids = torch.norm(pos[i:i+1] - centroids, p=2, dim=1)
                group_indices[b, i] = dists_to_centroids.argmin()
    
    return group_indices


class GraphBuilder(nn.Module):
    """Neural module for building and managing group-aware graphs"""
    
    def __init__(self, 
                 distance_scales: List[float] = [0.5, 1.0, 2.0],
                 group_threshold: float = 2.0,
                 group_method: str = 'threshold'):
        super().__init__()
        self.distance_scales = distance_scales
        self.group_threshold = group_threshold
        self.group_method = group_method
        
    def forward(self, positions: torch.Tensor, 
                displacements: torch.Tensor,
                pairwise_dist: Optional[torch.Tensor] = None,
                pairwise_disp: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build group-aware graphs
        
        Args:
            positions: (B, T, N, 2) - agent positions
            displacements: (B, T, N, 2) - agent displacements  
            pairwise_dist: (B, T, N, N) - precomputed distances (optional)
            pairwise_disp: (B, T, N, N) - precomputed displacements (optional)
            
        Returns:
            adjacency: (B, R, T, N, N) - multi-relational adjacency
            group_indices: (B, N) - group assignments
            intra_adjacency: (B, R, T, N, N) - intra-group adjacency
        """
        B, T, N, _ = positions.shape
        
        # Compute pairwise matrices if not provided
        if pairwise_dist is None:
            pairwise_dist = torch.zeros(B, T, N, N, device=positions.device)
            for b in range(B):
                for t in range(T):
                    for i in range(N):
                        for j in range(N):
                            if i != j:
                                pairwise_dist[b, t, i, j] = torch.norm(
                                    positions[b, t, i] - positions[b, t, j], p=2)
        
        if pairwise_disp is None:
            pairwise_disp = torch.zeros(B, T, N, N, device=displacements.device)
            for b in range(B):
                for t in range(T):
                    for i in range(N):
                        for j in range(N):
                            if i != j:
                                pairwise_disp[b, t, i, j] = torch.norm(
                                    displacements[b, t, i] - displacements[b, t, j], p=2)
        
        # Build multi-relational adjacency
        adjacency = build_relations(pairwise_dist, pairwise_disp, self.distance_scales)
        
        # Assign groups
        group_indices = assign_groups_by_distance(
            positions, self.group_threshold, self.group_method)
        
        # Create intra-group adjacency
        intra_adjacency = mask_by_group(adjacency, group_indices)
        
        return adjacency, group_indices, intra_adjacency
