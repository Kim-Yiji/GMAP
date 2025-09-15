# DMRGCN Encoder Adapter
# Wraps DMRGCN to enforce unified [B, T, N, d] shape convention

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import logging

try:
    from einops import rearrange
except ImportError:
    # Fallback implementation if einops not available
    def rearrange(tensor, pattern, **axes_lengths):
        """Simple fallback for basic rearrange operations"""
        if pattern == 'b t n d -> b d t n':
            return tensor.permute(0, 3, 1, 2)
        elif pattern == 'b d t n -> b t n d':
            return tensor.permute(0, 2, 3, 1)
        elif pattern == 'b t n d -> b n t d':
            return tensor.permute(0, 2, 1, 3)
        elif pattern == 'b n t d -> b t n d':
            return tensor.permute(0, 2, 1, 3)
        else:
            raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")

from utils.shapes import (
    assert_shape, log_shape, mask_features, mask_adj, 
    ensure_time_axis, ensure_relation_axis, checked_shape
)
from model.backbone import DMRGCNBackbone

logger = logging.getLogger(__name__)


class DMRGCNEncoderAdapter(nn.Module):
    """Adapter for DMRGCN backbone with unified shape interface
    
    Enforces strict shape conventions:
    - Input: X [B, T_obs, N, d_in], A [B, T_obs, N, N], M [B, T_obs, N]
    - Output: H [B, T_obs, N, d_h]
    
    Handles internal channel-first conversions transparently.
    """
    
    def __init__(self, 
                 d_in: int = 2,
                 d_h: int = 128,
                 hidden_dims: list = [64, 64, 64, 64, 128],
                 kernel_size: Tuple[int, int] = (3, 1),
                 dropout: float = 0.1,
                 distance_scales: list = [0.5, 1.0, 2.0],
                 use_mdn: bool = False,
                 share_backbone: bool = True):
        """Initialize DMRGCN encoder adapter
        
        Args:
            d_in: Input feature dimension
            d_h: Output feature dimension  
            hidden_dims: Hidden layer dimensions for backbone
            kernel_size: (temporal, spatial) kernel size
            dropout: Dropout rate
            distance_scales: Distance scales for multi-relational graphs
            use_mdn: Whether to use mixture density networks
            share_backbone: Whether to share backbone across relation types
        """
        super().__init__()
        
        self.d_in = d_in
        self.d_h = d_h
        self.distance_scales = distance_scales
        self.num_relations = len(distance_scales)
        
        # Input projection if needed
        if d_in != hidden_dims[0]:
            self.input_proj = nn.Linear(d_in, hidden_dims[0])
        else:
            self.input_proj = nn.Identity()
        
        # Output projection to ensure d_h
        if hidden_dims[-1] != d_h:
            self.output_proj = nn.Linear(hidden_dims[-1], d_h)
        else:
            self.output_proj = nn.Identity()
        
        # DMRGCN backbone
        self.backbone = DMRGCNBackbone(
            input_channels=hidden_dims[0],
            hidden_channels=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout,
            relation=len(distance_scales),
            use_mdn=use_mdn
        )
        
        logger.info(f"DMRGCN Adapter: {d_in} -> {d_h}, relations: {self.num_relations}")

    @checked_shape
    def forward(self, 
                X: torch.Tensor,           # [B, T_obs, N, d_in]
                A: torch.Tensor,           # [B, T_obs, N, N] or [B, T_obs, R, N, N]
                M: torch.Tensor            # [B, T_obs, N]
                ) -> torch.Tensor:         # [B, T_obs, N, d_h]
        """Forward pass through DMRGCN encoder
        
        Args:
            X: Input features [B, T_obs, N, d_in]
            A: Adjacency matrices [B, T_obs, N, N] or [B, T_obs, R, N, N]
            M: Validity mask [B, T_obs, N]
            
        Returns:
            H: Encoded features [B, T_obs, N, d_h]
        """
        # Validate input shapes
        assert_shape(X, 'b t n d', 'input_features')
        assert_shape(M, 'b t n', 'mask')
        
        B, T_obs, N, d_in = X.shape
        assert d_in == self.d_in, f"Expected d_in={self.d_in}, got {d_in}"
        
        # Ensure adjacency has correct shape
        if A.dim() == 4:
            assert_shape(A, 'b t n n', 'adjacency')
            # Add relation dimension if needed
            A = ensure_relation_axis(A, self.num_relations)
        elif A.dim() == 5:
            assert_shape(A, 'b t r n n', 'multi_relation_adjacency')
        else:
            raise ValueError(f"Invalid adjacency shape: {A.shape}")
        
        log_shape("dmrgcn_adapter_input", X=X, A=A, M=M)
        
        # Apply masks early
        X_masked = mask_features(X, M)
        A_masked = mask_adj(A, M)
        
        # Input projection
        X_proj = self.input_proj(X_masked)  # [B, T_obs, N, hidden_dims[0]]
        log_shape("after_input_proj", X_proj=X_proj)
        
        # Convert to channel-first for DMRGCN backbone: [B, T, N, d] -> [B, d, T, N]
        X_backbone = rearrange(X_proj, 'b t n d -> b d t n')
        
        # Transpose adjacency for backbone: [B, T, R, N, N] -> [B, R, T, N, N]
        A_backbone = A_masked.permute(0, 2, 1, 3, 4)
        
        log_shape("backbone_input", X_backbone=X_backbone, A_backbone=A_backbone)
        
        # Forward through DMRGCN backbone
        H_backbone = self.backbone(X_backbone, A_backbone)  # [B, d_out, T, N]
        
        log_shape("backbone_output", H_backbone=H_backbone)
        
        # Convert back to [B, T, N, d] format
        H = rearrange(H_backbone, 'b d t n -> b t n d')
        
        # Output projection to ensure d_h
        H = self.output_proj(H)  # [B, T_obs, N, d_h]
        
        # Final masking to ensure invalid positions are zero
        H = mask_features(H, M)
        
        # Validate output shape
        assert_shape(H, 'b t n d', 'output_features')
        assert H.shape[-1] == self.d_h, f"Output dimension mismatch: got {H.shape[-1]}, expected {self.d_h}"
        
        log_shape("dmrgcn_adapter_output", H=H)
        return H

    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.d_h


class MultiModalDMRGCNAdapter(nn.Module):
    """Multi-modal DMRGCN adapter supporting different input types
    
    Can handle:
    - Position features: [B, T, N, 2] 
    - Velocity features: [B, T, N, 2]
    - Combined features: [B, T, N, 4]
    """
    
    def __init__(self,
                 position_dim: int = 2,
                 velocity_dim: int = 2, 
                 d_h: int = 128,
                 use_velocity: bool = True,
                 **dmrgcn_kwargs):
        """Initialize multi-modal adapter
        
        Args:
            position_dim: Position feature dimension
            velocity_dim: Velocity feature dimension
            d_h: Output hidden dimension
            use_velocity: Whether to use velocity features
            **dmrgcn_kwargs: Arguments for DMRGCN adapter
        """
        super().__init__()
        
        self.position_dim = position_dim
        self.velocity_dim = velocity_dim
        self.use_velocity = use_velocity
        
        # Calculate total input dimension
        total_dim = position_dim
        if use_velocity:
            total_dim += velocity_dim
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, d_h),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_h),
            nn.Linear(d_h, d_h)
        )
        
        # DMRGCN adapter
        self.dmrgcn = DMRGCNEncoderAdapter(
            d_in=d_h,
            d_h=d_h,
            hidden_dims=[d_h, d_h, d_h],
            **dmrgcn_kwargs
        )

    def compute_velocities(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Compute velocities from positions
        
        Args:
            X: Positions [B, T, N, 2]
            M: Mask [B, T, N]
            
        Returns:
            V: Velocities [B, T, N, 2]
        """
        B, T, N, _ = X.shape
        V = torch.zeros_like(X)
        
        # Compute finite differences
        V[:, 1:] = X[:, 1:] - X[:, :-1]  # [B, T-1, N, 2]
        
        # Mask invalid transitions
        valid_transitions = M[:, :-1] & M[:, 1:]  # [B, T-1, N]
        V[:, 1:] = V[:, 1:] * valid_transitions.unsqueeze(-1)
        
        return V

    @checked_shape
    def forward(self,
                X_pos: torch.Tensor,      # [B, T, N, 2] - positions
                A: torch.Tensor,          # [B, T, N, N] or [B, T, R, N, N]
                M: torch.Tensor,          # [B, T, N]
                X_vel: Optional[torch.Tensor] = None  # [B, T, N, 2] - velocities (optional)
                ) -> torch.Tensor:        # [B, T, N, d_h]
        """Forward pass with multi-modal features
        
        Args:
            X_pos: Position features [B, T, N, 2]
            A: Adjacency matrices
            M: Validity mask
            X_vel: Velocity features (computed if None)
            
        Returns:
            H: Encoded features [B, T, N, d_h]
        """
        assert_shape(X_pos, 'b t n d', 'positions')
        assert X_pos.shape[-1] == self.position_dim, f"Position dim mismatch"
        
        features = [X_pos]
        
        if self.use_velocity:
            if X_vel is None:
                X_vel = self.compute_velocities(X_pos, M)
            else:
                assert_shape(X_vel, 'b t n d', 'velocities')
                assert X_vel.shape[-1] == self.velocity_dim, f"Velocity dim mismatch"
            
            features.append(X_vel)
        
        # Concatenate features
        X_combined = torch.cat(features, dim=-1)  # [B, T, N, total_dim]
        
        # Feature fusion
        X_fused = self.feature_fusion(X_combined)  # [B, T, N, d_h]
        
        # Apply masking
        X_fused = mask_features(X_fused, M)
        
        # Forward through DMRGCN
        H = self.dmrgcn(X_fused, A, M)
        
        return H
