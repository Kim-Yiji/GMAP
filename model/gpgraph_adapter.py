# GP-Graph adapter components for group-aware trajectory prediction
# Integrates group assignment and feature integration modules

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GroupAssignment(nn.Module):
    """Group assignment module with multiple strategies for pedestrian grouping
    
    Based on GP-Graph GroupGenerator with enhancements for integration with DMRGCN.
    """
    
    def __init__(self, d_type='learned', th=1.0, in_channels=16, 
                 hid_channels=32, n_head=1, dropout=0, st_estimator=False):
        super().__init__()
        self.d_type = d_type
        self.st_estimator = st_estimator  # Use spatio-temporal features for estimation
        
        # Group assignment networks
        if d_type == 'learned':
            self.group_cnn = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, 1),
                nn.ReLU(),
                nn.BatchNorm2d(hid_channels),
                nn.Dropout(dropout, inplace=True),
                nn.Conv2d(hid_channels, n_head, 1),
            )
        elif d_type == 'estimate_th':
            self.group_cnn = nn.Sequential(
                nn.Conv2d(in_channels, n_head, 1),
            )
        elif d_type == 'learned_l2norm':
            self.group_cnn = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 1), padding=(1, 0))
            )
        elif d_type == 'euclidean':
            # No learnable parameters for euclidean distance
            self.group_cnn = None
        
        # Threshold parameter
        self.th = th if isinstance(th, float) else nn.Parameter(torch.Tensor([th]))
        
        # Optional spatio-temporal feature encoder
        if st_estimator:
            self.st_encoder = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=(3, 1), padding=(1, 0)),  # 2D positions -> features
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
                nn.ReLU(),
                nn.Conv2d(32, in_channels, kernel_size=1)  # Match input channels
            )

    def find_group_indices(self, v, dist_mat):
        """Find group indices using threshold-based clustering"""
        n_ped = v.size(-1)
        device = v.device
        
        # Handle edge case: single agent
        if n_ped == 1:
            return torch.tensor([0], dtype=torch.long, device=device)
        
        # Create mask for upper triangular matrix
        mask = torch.ones_like(dist_mat).mul(1e4).triu()
        
        # Find pairs closer than threshold
        close_condition = dist_mat.tril(diagonal=-1).add(mask).le(self.th)
        close_pairs = torch.nonzero(close_condition, as_tuple=True)
        
        # Initialize group indices
        indices_raw = torch.arange(n_ped, dtype=torch.long, device=device)
        
        # Union-find like grouping - fix aliasing issue
        if len(close_pairs[0]) > 0:  # Only process if there are close pairs
            for r, c in zip(close_pairs[0], close_pairs[1]):
                # Boundary check to prevent index errors
                if r.item() < n_ped and c.item() < n_ped:
                    # Merge groups - clone to avoid aliasing
                    r_val = indices_raw[r].clone()  # Store value to avoid aliasing
                    mask_merge = indices_raw == r_val
                    c_val = indices_raw[c].clone()  # Store value to avoid aliasing
                    indices_raw = indices_raw.clone()  # Clone entire tensor to break aliasing
                    indices_raw[mask_merge] = c_val
        
        # Compress indices to contiguous range
        indices_unique = indices_raw.unique()
        indices_map = torch.arange(indices_unique.size(0), dtype=torch.long, device=device)
        indices = torch.zeros_like(indices_raw)
        
        for i, j in zip(indices_unique, indices_map):
            indices[indices_raw == i] = j
        
        return indices

    def compute_distance_matrix(self, v_abs, v_rel=None):
        """Compute distance matrix using specified method"""
        n_ped = v_abs.size(-1)
        
        if self.d_type == 'euclidean':
            # Use absolute positions for euclidean distance
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
            
        elif self.d_type == 'learned_l2norm':
            # Learned feature space distance
            temp = self.group_cnn(v_abs).unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
            
        elif self.d_type == 'learned':
            # Fully learned distance
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-1, -2)).reshape(temp.size(0), -1, n_ped, n_ped)
            temp = self.group_cnn(temp).exp()
            # Make symmetric
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1)
            
        elif self.d_type == 'estimate_th':
            # Estimate threshold dynamically
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-2, -1))
            dist_mat = temp.norm(p=2, dim=1)
            # Update threshold
            self.th = self.group_cnn(temp.reshape(temp.size(0), -1, n_ped, n_ped)).mean().exp()
            
        else:
            raise NotImplementedError(f"Distance type {self.d_type} not implemented")
        
        return dist_mat

    def group_backprop_trick(self, v, dist_mat, tau=1.0, hard=False):
        """Differentiable grouping using straight-through estimator"""
        # Soft assignment probabilities
        sig = (-(dist_mat - self.th) / tau).sigmoid()
        sig_norm = sig / (sig.sum(dim=0, keepdim=True) + 1e-8)
        
        # Soft grouping
        v_soft = v @ sig_norm
        
        if hard:
            # Hard assignment with straight-through gradient
            return (v - v_soft).detach() + v_soft
        else:
            return v_soft

    def forward(self, v_rel, v_abs=None, tau=0.1, hard=True):
        """
        Args:
            v_rel: (B, C, T, N) - relative displacement features
            v_abs: (B, 2, T, N) - absolute position features (optional)
            tau: temperature for soft assignment
            hard: whether to use hard assignment
            
        Returns:
            v_grouped: (B, C, T, N) - grouped features
            group_indices: (B, N) - group assignments
            dist_mat: (B, T, N, N) - distance matrices
        """
        assert v_rel.size(0) == 1, "Currently supports batch size 1"
        
        B, C, T, N = v_rel.shape
        
        # Use absolute positions if provided, otherwise use relative features
        if v_abs is None:
            v_abs = v_rel[:, :2, :, :] if C >= 2 else v_rel
        
        # Apply ST estimator if enabled
        if self.st_estimator and hasattr(self, 'st_encoder'):
            v_abs = self.st_encoder(v_abs)
        
        # Compute distance matrix (use last timestep for grouping)
        v_abs_last = v_abs[:, :, -1:, :]  # (B, C, 1, N)
        dist_mat = self.compute_distance_matrix(v_abs_last)  # (B, 1, N, N)
        dist_mat = dist_mat.squeeze(1).mean(dim=0)  # (N, N) - average over batch and time
        
        # Find group indices
        group_indices = self.find_group_indices(v_rel, dist_mat)  # (N,)
        group_indices = group_indices.unsqueeze(0)  # (B, N)
        
        # Apply differentiable grouping
        v_grouped = self.group_backprop_trick(
            v_rel.squeeze(0), dist_mat, tau=tau, hard=hard
        ).unsqueeze(0)  # (B, C, T, N)
        
        # Expand distance matrix for output
        dist_mat_full = dist_mat.unsqueeze(0).unsqueeze(1).expand(B, T, -1, -1)  # (B, T, N, N)
        
        # Save group assignment data
        result_dict = {
            'grouped_features': v_grouped.cpu().numpy(),
            'group_indices': group_indices.cpu().numpy(),
            'distance_matrix': dist_mat_full.cpu().numpy(),
            'timestamp': torch.tensor([v_abs.size(2)]).numpy(),  # Save temporal context
        }
        
        return v_grouped, group_indices, dist_mat_full, result_dict


class GroupIntegration(nn.Module):
    """Feature integration module for combining multi-path features
    
    Combines features from agent-level, intra-group, and inter-group processing.
    """
    
    def __init__(self, mix_type='mean', n_mix=3, out_channels=5, 
                 pred_seq_len=12, feature_dim=64):
        super().__init__()
        self.mix_type = mix_type
        self.pred_seq_len = pred_seq_len
        self.n_mix = n_mix
        
        if mix_type == 'mlp':
            # MLP-based fusion
            self.fusion_mlp = nn.Sequential(
                nn.Conv2d(out_channels * pred_seq_len * n_mix, 
                         out_channels * pred_seq_len * 2, kernel_size=1),
                nn.PReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(out_channels * pred_seq_len * 2, 
                         out_channels * pred_seq_len, kernel_size=1)
            )
            
        elif mix_type == 'cnn':
            # CNN-based fusion
            self.fusion_cnn = nn.Sequential(
                nn.Conv2d(out_channels * n_mix, out_channels * 2, 
                         kernel_size=(3, 1), padding=(1, 0)),
                nn.PReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(out_channels * 2, out_channels, 
                         kernel_size=(3, 1), padding=(1, 0))
            )
            
        elif mix_type == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True
            )
            self.norm = nn.LayerNorm(feature_dim)
            
        elif mix_type == 'concat_mlp':
            # Simple concatenation + MLP
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(out_channels * n_mix, out_channels * 2, kernel_size=1),
                nn.PReLU(),
                nn.Dropout(0.1),
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
            )

    def forward(self, feature_list):
        """
        Args:
            feature_list: List of (B, C, T, N) feature tensors from different paths
            
        Returns:
            fused_features: (B, C, T, N) - integrated features
        """
        if len(feature_list) != self.n_mix:
            raise ValueError(f"Expected {self.n_mix} features, got {len(feature_list)}")
        
        B, C, T, N = feature_list[0].shape
        
        if self.mix_type == 'sum':
            # Simple summation
            fused = torch.stack(feature_list, dim=0).sum(dim=0)
            
        elif self.mix_type == 'mean':
            # Simple averaging
            fused = torch.stack(feature_list, dim=0).mean(dim=0)
            
        elif self.mix_type == 'mlp':
            # MLP-based fusion
            # Concatenate and reshape for MLP
            concat_features = torch.cat(feature_list, dim=1)  # (B, C*n_mix, T, N)
            concat_features = concat_features.reshape(B, -1, 1, N)  # (B, C*T*n_mix, 1, N)
            fused = self.fusion_mlp(concat_features)  # (B, C*T, 1, N)
            fused = fused.view(B, C, T, N)  # Reshape back
            
        elif self.mix_type == 'cnn':
            # CNN-based fusion
            concat_features = torch.cat(feature_list, dim=1)  # (B, C*n_mix, T, N)
            residual = torch.stack(feature_list, dim=0).mean(dim=0)  # Residual connection
            fused = self.fusion_cnn(concat_features) + residual
            
        elif self.mix_type == 'attention':
            # Attention-based fusion
            # Reshape for attention: (B*N, T, C)
            stacked = torch.stack(feature_list, dim=1)  # (B, n_mix, C, T, N)
            stacked = stacked.permute(0, 4, 3, 1, 2)  # (B, N, T, n_mix, C)
            stacked = stacked.reshape(B * N, T, self.n_mix * C)
            
            # Apply self-attention
            attended, _ = self.attention(stacked, stacked, stacked)
            attended = self.norm(attended + stacked)
            
            # Reshape back: (B, C, T, N)
            attended = attended.reshape(B, N, T, self.n_mix * C)
            attended = attended.permute(0, 3, 2, 1)  # (B, n_mix*C, T, N)
            fused = attended[:, :C, :, :]  # Take first C channels
            
        elif self.mix_type == 'concat_mlp':
            # Concatenation + MLP
            concat_features = torch.cat(feature_list, dim=1)  # (B, C*n_mix, T, N)
            fused = self.concat_mlp(concat_features)
            
        else:
            raise NotImplementedError(f"Mix type {self.mix_type} not implemented")
        
        return fused


def ped_group_pool(features, group_indices):
    """Pool agent features to group level
    
    Args:
        features: (B, C, T, N) - agent-level features
        group_indices: (B, N) - group assignments
        
    Returns:
        group_features: (B, C, T, Ng) - group-level features
    """
    B, C, T, N = features.shape
    device = features.device
    
    # Get number of groups
    max_group = group_indices.max().item() + 1
    
    # Initialize group features
    group_features = torch.zeros(B, C, T, max_group, device=device, dtype=features.dtype)
    group_counts = torch.zeros(B, max_group, device=device, dtype=torch.float)
    
    # Pool features by group
    for b in range(B):
        for g in range(max_group):
            mask = (group_indices[b] == g)
            if mask.sum() > 0:
                group_features[b, :, :, g] = features[b, :, :, mask].mean(dim=-1)
                group_counts[b, g] = mask.sum().float()
    
    return group_features


def ped_group_unpool(group_features, group_indices):
    """Unpool group features back to agent level
    
    Args:
        group_features: (B, C, T, Ng) - group-level features
        group_indices: (B, N) - group assignments
        
    Returns:
        agent_features: (B, C, T, N) - agent-level features
    """
    B, C, T, Ng = group_features.shape
    N = group_indices.shape[1]
    device = group_features.device
    
    # Initialize agent features
    agent_features = torch.zeros(B, C, T, N, device=device, dtype=group_features.dtype)
    
    # Assign group features to agents
    for b in range(B):
        for n in range(N):
            g = group_indices[b, n].item()
            agent_features[b, :, :, n] = group_features[b, :, :, g]
    
    return agent_features


def ped_group_mask(group_indices):
    """Create mask for intra-group connections
    
    Args:
        group_indices: (B, N) or (N,) - group assignments
        
    Returns:
        mask: (N, N) - boolean mask for same-group connections
    """
    if group_indices.dim() == 2:
        group_indices = group_indices.squeeze(0)  # Assume batch size 1
    
    N = group_indices.size(0)
    device = group_indices.device
    
    # Create identity mask
    mask = torch.eye(N, dtype=torch.bool, device=device)
    
    # Add same-group connections
    for group_id in group_indices.unique():
        group_mask = (group_indices == group_id)
        indices = torch.nonzero(group_mask, as_tuple=False).squeeze(1)
        
        # Set all pairs within group to True
        for i in indices:
            mask[i, indices] = True
    
    return mask
