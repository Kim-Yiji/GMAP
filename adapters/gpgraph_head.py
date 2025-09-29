# GP-Graph Head Adapter
# Wraps GP-Graph prediction head with unified shape interface

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
import logging

try:
    from einops import rearrange, repeat
except ImportError:
    def rearrange(tensor, pattern, **axes_lengths):
        """Fallback for basic patterns"""
        if pattern == 'b n d -> b 1 n d':
            return tensor.unsqueeze(1)
        elif pattern == 'b 1 n d -> b n d':
            return tensor.squeeze(1)
        else:
            raise NotImplementedError(f"Unsupported pattern: {pattern}")
    
    def repeat(tensor, pattern, **axes_lengths):
        """Fallback for basic repeat patterns"""
        if pattern == 'b 1 n n -> b t n n':
            t = axes_lengths['t']
            return tensor.expand(-1, t, -1, -1)
        elif pattern == 'b 1 n1 n2 -> b t n1 n2':
            t = axes_lengths['t']
            return tensor.expand(-1, t, -1, -1)
        else:
            raise NotImplementedError(f"Unsupported repeat pattern: {pattern}")

from utils.shapes import (
    assert_shape, log_shape, mask_features, aggregate_time_features,
    ensure_time_axis, checked_shape
)
from model.gpgraph_adapter import GroupAssignment, GroupIntegration

logger = logging.getLogger(__name__)


class GPGraphHead(nn.Module):
    """GP-Graph prediction head with unified shape interface
    
    Input: H [B, T_obs, N, d_h], A_pred [B, T_pred, N, N] (optional), M_pred [B, T_pred, N]
    Output: ΔY [B, T_pred, N, 2]
    
    Handles temporal aggregation and group-aware prediction.
    """
    
    def __init__(self,
                 d_h: int = 128,
                 d_gp_in: int = 128,
                 T_pred: int = 12,
                 agg_method: str = 'last',
                 group_type: str = 'euclidean',
                 group_threshold: float = 2.0,
                 mix_type: str = 'mean',
                 enable_paths: Dict[str, bool] = None,
                 output_dim: int = 2):
        """Initialize GP-Graph head
        
        Args:
            d_h: Input hidden dimension from encoder
            d_gp_in: GP-Graph input dimension
            T_pred: Prediction sequence length
            agg_method: Time aggregation method ('last', 'mean', 'gru')
            group_type: Group assignment type
            group_threshold: Distance threshold for grouping
            mix_type: Feature fusion method
            enable_paths: Which processing paths to enable
            output_dim: Output dimension (2 for x,y deltas)
        """
        super().__init__()
        
        self.d_h = d_h
        self.d_gp_in = d_gp_in
        self.T_pred = T_pred
        self.agg_method = agg_method
        self.output_dim = output_dim
        
        if enable_paths is None:
            enable_paths = {'agent': True, 'intra': True, 'inter': True}
        self.enable_paths = enable_paths
        self.n_paths = sum(enable_paths.values())
        
        # Feature dimension alignment
        if d_h != d_gp_in:
            self.feature_proj = nn.Linear(d_h, d_gp_in)
        else:
            self.feature_proj = nn.Identity()
        
        # Temporal aggregation modules
        if agg_method == 'gru':
            self.temporal_gru = nn.GRU(
                input_size=d_gp_in,
                hidden_size=d_gp_in,
                num_layers=1,
                batch_first=True
            )
        
        # Group assignment module
        self.group_assignment = GroupAssignment(
            d_type=group_type,
            th=group_threshold,
            in_channels=d_gp_in,
            hid_channels=64,
            st_estimator=True
        )
        
        # Feature integration
        self.feature_integration = GroupIntegration(
            mix_type=mix_type,
            n_mix=self.n_paths,
            out_channels=d_gp_in,
            pred_seq_len=T_pred
        )
        
        # Prediction heads for different paths
        self.prediction_heads = nn.ModuleDict()
        for path_name in ['agent', 'intra', 'inter']:
            if enable_paths.get(path_name, False):
                self.prediction_heads[path_name] = self._create_prediction_head(d_gp_in)
        
        # Final output projection
        self.output_head = nn.Sequential(
            nn.Linear(d_gp_in, d_gp_in // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_gp_in // 2),
            nn.Linear(d_gp_in // 2, output_dim * T_pred)
        )
        
        logger.info(f"GP-Graph Head: {d_h} -> {d_gp_in} -> {output_dim}, T_pred={T_pred}")

    def _create_prediction_head(self, input_dim: int) -> nn.Module:
        """Create prediction head for specific path"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )

    @checked_shape  
    def aggregate_temporal_features(self, 
                                  H: torch.Tensor,     # [B, T_obs, N, d_h]
                                  M_obs: torch.Tensor  # [B, T_obs, N]
                                  ) -> torch.Tensor:   # [B, N, d_gp_in]
        """Aggregate features across observation time
        
        Args:
            H: Encoded features [B, T_obs, N, d_h]
            M_obs: Observation mask [B, T_obs, N]
            
        Returns:
            H_agg: Aggregated features [B, N, d_gp_in]
        """
        assert_shape(H, 'b t n d', 'encoded_features')
        assert_shape(M_obs, 'b t n', 'obs_mask')
        
        B, T_obs, N, d_h = H.shape
        
        # Project features to GP-Graph input dimension
        H_proj = self.feature_proj(H)  # [B, T_obs, N, d_gp_in]
        
        if self.agg_method == 'last':
            H_agg = aggregate_time_features(H_proj, M_obs, method='last')
            
        elif self.agg_method == 'mean':
            H_agg = aggregate_time_features(H_proj, M_obs, method='mean')
            
        elif self.agg_method == 'gru':
            # Reshape for GRU: [B*N, T_obs, d_gp_in]
            H_gru = H_proj.view(B * N, T_obs, self.d_gp_in)
            
            # Create sequence lengths for packing
            seq_lengths = M_obs.sum(dim=1).view(-1).cpu()  # [B*N]
            
            # Forward through GRU
            if seq_lengths.min() > 0:  # All sequences have at least one valid timestep
                H_gru_out, _ = self.temporal_gru(H_gru)  # [B*N, T_obs, d_gp_in]
                
                # Take last valid output for each sequence
                last_indices = (seq_lengths - 1).clamp(min=0).long()
                batch_indices = torch.arange(B * N, device=H.device)
                H_agg = H_gru_out[batch_indices, last_indices]  # [B*N, d_gp_in]
                H_agg = H_agg.view(B, N, self.d_gp_in)
            else:
                # Fallback for invalid sequences
                H_agg = H_proj[:, -1]  # [B, N, d_gp_in]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")
        
        log_shape("temporal_aggregation", H=H, H_proj=H_proj, H_agg=H_agg)
        return H_agg

    def create_prediction_adjacency(self, 
                                  A_obs: torch.Tensor,       # [B, T_obs, N, N]
                                  M_pred: torch.Tensor       # [B, T_pred, N]
                                  ) -> torch.Tensor:         # [B, T_pred, N, N]
        """Create adjacency matrices for prediction phase
        
        Args:
            A_obs: Observed adjacency [B, T_obs, N, N]
            M_pred: Prediction mask [B, T_pred, N]
            
        Returns:
            A_pred: Prediction adjacency [B, T_pred, N, N]
        """
        # Use last observed adjacency and replicate for prediction horizon
        A_last = A_obs[:, -1:, :, :]  # [B, 1, N, N]
        A_pred = repeat(A_last, 'b 1 n1 n2 -> b t n1 n2', t=self.T_pred)
        
        # Apply prediction mask
        M_i = M_pred.unsqueeze(-1)      # [B, T_pred, N, 1]
        M_j = M_pred.unsqueeze(-2)      # [B, T_pred, 1, N]
        A_pred = A_pred * M_i * M_j     # Mask edges where either node is invalid
        
        return A_pred

    @checked_shape
    def forward(self,
                H: torch.Tensor,                        # [B, T_obs, N, d_h]
                A_obs: torch.Tensor,                    # [B, T_obs, N, N]
                M_obs: torch.Tensor,                    # [B, T_obs, N]
                M_pred: torch.Tensor,                   # [B, T_pred, N]
                A_pred: Optional[torch.Tensor] = None,  # [B, T_pred, N, N] (optional)
                return_groups: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through GP-Graph head
        
        Args:
            H: Encoded features [B, T_obs, N, d_h]
            A_obs: Observed adjacency [B, T_obs, N, N]
            M_obs: Observation mask [B, T_obs, N]
            M_pred: Prediction mask [B, T_pred, N]
            A_pred: Prediction adjacency (optional)
            return_groups: Whether to return group assignments
            
        Returns:
            ΔY: Predicted deltas [B, T_pred, N, 2]
            group_indices: Group assignments [B, N] (if return_groups=True)
        """
        # Validate input shapes
        assert_shape(H, 'b t n d', 'encoded_features')
        assert_shape(A_obs, 'b t n n', 'observed_adjacency')
        assert_shape(M_obs, 'b t n', 'observed_mask')
        assert_shape(M_pred, 'b t n', 'prediction_mask')
        
        B, T_obs, N, d_h = H.shape
        T_pred = M_pred.shape[1]
        assert T_pred == self.T_pred, f"Prediction length mismatch: got {T_pred}, expected {self.T_pred}"
        
        log_shape("gpgraph_head_input", H=H, A_obs=A_obs, M_obs=M_obs, M_pred=M_pred)
        
        # Temporal aggregation
        H_agg = self.aggregate_temporal_features(H, M_obs)  # [B, N, d_gp_in]
        
        # Create prediction adjacency if not provided
        if A_pred is None:
            A_pred = self.create_prediction_adjacency(A_obs, M_pred)
        else:
            assert_shape(A_pred, 'b t n n', 'prediction_adjacency')
        
        # Group assignment based on aggregated features
        # Convert to format expected by GroupAssignment: [B, C, T, N]
        H_for_grouping = rearrange(H_agg, 'b n d -> b d 1 n')
        
        # Get positions for grouping (assume first 2 dims are x,y if available)
        if self.d_gp_in >= 2:
            pos_features = H_agg[:, :, :2].unsqueeze(1)  # [B, 1, N, 2]
            pos_for_grouping = rearrange(pos_features, 'b t n d -> b d t n')
        else:
            pos_for_grouping = H_for_grouping
        
        # Perform group assignment
        _, group_indices, _ = self.group_assignment(
            H_for_grouping, pos_for_grouping, hard=True
        )
        group_indices = group_indices.squeeze(0) if group_indices.dim() == 2 else group_indices[0]
        
        log_shape("group_assignment", H_agg=H_agg, group_indices=group_indices)
        
        # Multi-path processing (simplified for this adapter)
        path_features = []
        
        if self.enable_paths['agent']:
            # Agent-level processing
            feat_agent = self.prediction_heads['agent'](H_agg)
            path_features.append(feat_agent.unsqueeze(2).expand(-1, -1, T_pred, -1))
        
        if self.enable_paths['intra']:
            # Intra-group processing (simplified)
            feat_intra = self.prediction_heads['intra'](H_agg)
            path_features.append(feat_intra.unsqueeze(2).expand(-1, -1, T_pred, -1))
        
        if self.enable_paths['inter']:
            # Inter-group processing (simplified)
            feat_inter = self.prediction_heads['inter'](H_agg)
            path_features.append(feat_inter.unsqueeze(2).expand(-1, -1, T_pred, -1))
        
        # Feature integration
        if len(path_features) > 1:
            # Convert to format expected by integration: [B, C, T, N]
            path_features_integrated = []
            for feat in path_features:
                feat_integrated = rearrange(feat, 'b n t d -> b d t n')
                path_features_integrated.append(feat_integrated)
            
            # Integrate features
            fused_features = self.feature_integration(path_features_integrated)  # [B, d, T, N]
            fused_features = rearrange(fused_features, 'b d t n -> b t n d')  # [B, T_pred, N, d]
        else:
            fused_features = path_features[0].permute(0, 2, 1, 3)  # [B, T_pred, N, d]
        
        # Apply prediction mask
        fused_features = mask_features(fused_features, M_pred)
        
        # Generate predictions for each timestep
        delta_Y = torch.zeros(B, T_pred, N, self.output_dim, device=H.device, dtype=H.dtype)
        
        for t in range(T_pred):
            # Use features at current timestep
            feat_t = fused_features[:, t, :, :]  # [B, N, d_gp_in]
            
            # Generate predictions
            pred_t = self.output_head(feat_t)  # [B, N, output_dim * T_pred]
            pred_t = pred_t.view(B, N, self.output_dim, T_pred)  # [B, N, output_dim, T_pred]
            
            # Take prediction for current timestep
            delta_Y[:, t, :, :] = pred_t[:, :, :, t]
        
        # Apply final masking
        delta_Y = mask_features(delta_Y, M_pred)
        
        # Validate output shape
        assert_shape(delta_Y, 'b t n d', 'predicted_deltas')
        assert delta_Y.shape[-1] == self.output_dim, f"Output dimension mismatch"
        
        log_shape("gpgraph_head_output", delta_Y=delta_Y)
        
        if return_groups:
            return delta_Y, group_indices
        else:
            return delta_Y


class SimpleRegressionHead(nn.Module):
    """Simple regression head as fallback option
    
    Direct regression from aggregated features to trajectory deltas.
    """
    
    def __init__(self,
                 d_h: int = 128,
                 T_pred: int = 12,
                 output_dim: int = 2,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.d_h = d_h
        self.T_pred = T_pred
        self.output_dim = output_dim
        
        # Simple MLP for trajectory prediction
        self.predictor = nn.Sequential(
            nn.Linear(d_h, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim * T_pred)
        )

    @checked_shape
    def forward(self,
                H: torch.Tensor,        # [B, T_obs, N, d_h]
                M_obs: torch.Tensor,    # [B, T_obs, N]
                M_pred: torch.Tensor    # [B, T_pred, N]
                ) -> torch.Tensor:      # [B, T_pred, N, 2]
        """Simple regression forward pass"""
        
        # Aggregate temporal features (use last valid timestep)
        H_agg = aggregate_time_features(H, M_obs, method='last')  # [B, N, d_h]
        
        # Generate predictions
        pred_flat = self.predictor(H_agg)  # [B, N, output_dim * T_pred]
        pred_flat = pred_flat.view(H.shape[0], H.shape[2], self.output_dim, self.T_pred)
        delta_Y = pred_flat.permute(0, 3, 1, 2)  # [B, T_pred, N, output_dim]
        
        # Apply prediction mask
        delta_Y = mask_features(delta_Y, M_pred)
        
        return delta_Y
