# Unified DMRGCN + GP-Graph Model
# Main integration model with strict shape conventions

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, Union
import logging

from utils.shapes import (
    assert_shape, log_shape, mask_features, ensure_time_axis, 
    validate_model_io, checked_shape
)
from adapters.dmrgcn_adapter import DMRGCNEncoderAdapter, MultiModalDMRGCNAdapter  
from adapters.gpgraph_head import GPGraphHead, SimpleRegressionHead

logger = logging.getLogger(__name__)


class DMRGCN_GPGraph_Model(nn.Module):
    """Unified DMRGCN + GP-Graph model with strict shape conventions
    
    Enforces unified pipeline:
    Input:  X [B, T_obs, N, d_in], A_obs [B, T_obs, N, N], M_obs [B, T_obs, N]
    Output: ΔY [B, T_pred, N, 2]
    
    Internal flow:
    1. DMRGCN Encoder: X -> H [B, T_obs, N, d_h]
    2. GP-Graph Head: H -> ΔY [B, T_pred, N, 2]
    """
    
    def __init__(self,
                 # Input/output dimensions
                 d_in: int = 2,
                 d_h: int = 128,
                 d_gp_in: int = 128,
                 T_pred: int = 12,
                 output_dim: int = 2,
                 
                 # DMRGCN parameters
                 dmrgcn_hidden_dims: list = [64, 64, 64, 64, 128],
                 dmrgcn_kernel_size: Tuple[int, int] = (3, 1),
                 dmrgcn_dropout: float = 0.1,
                 distance_scales: list = [0.5, 1.0, 2.0],
                 
                 # GP-Graph parameters
                 agg_method: str = 'last',
                 group_type: str = 'euclidean',
                 group_threshold: float = 2.0,
                 mix_type: str = 'mean',
                 enable_paths: Dict[str, bool] = None,
                 
                 # Model configuration
                 use_multimodal: bool = False,
                 use_simple_head: bool = False,
                 share_backbone: bool = True):
        """Initialize unified model
        
        Args:
            d_in: Input feature dimension
            d_h: Hidden dimension from encoder
            d_gp_in: GP-Graph input dimension
            T_pred: Prediction sequence length
            output_dim: Output dimension (2 for x,y deltas)
            dmrgcn_*: DMRGCN-specific parameters
            agg_method: Temporal aggregation method
            group_type: Group assignment method
            group_threshold: Distance threshold for grouping
            mix_type: Feature fusion method
            enable_paths: Which processing paths to enable
            use_multimodal: Whether to use multi-modal encoder
            use_simple_head: Whether to use simple regression head
            share_backbone: Whether to share backbone across paths
        """
        super().__init__()
        
        self.d_in = d_in
        self.d_h = d_h
        self.d_gp_in = d_gp_in
        self.T_pred = T_pred
        self.output_dim = output_dim
        self.use_multimodal = use_multimodal
        self.use_simple_head = use_simple_head
        
        # Default path configuration
        if enable_paths is None:
            enable_paths = {'agent': True, 'intra': True, 'inter': True}
        self.enable_paths = enable_paths
        
        # Encoder selection
        if use_multimodal:
            self.encoder = MultiModalDMRGCNAdapter(
                position_dim=2,
                velocity_dim=2,
                d_h=d_h,
                use_velocity=True,
                hidden_dims=dmrgcn_hidden_dims,
                kernel_size=dmrgcn_kernel_size,
                dropout=dmrgcn_dropout,
                distance_scales=distance_scales
            )
        else:
            self.encoder = DMRGCNEncoderAdapter(
                d_in=d_in,
                d_h=d_h,
                hidden_dims=dmrgcn_hidden_dims,
                kernel_size=dmrgcn_kernel_size,
                dropout=dmrgcn_dropout,
                distance_scales=distance_scales
            )
        
        # Head selection
        if use_simple_head:
            self.head = SimpleRegressionHead(
                d_h=d_h,
                T_pred=T_pred,
                output_dim=output_dim
            )
        else:
            self.head = GPGraphHead(
                d_h=d_h,
                d_gp_in=d_gp_in,
                T_pred=T_pred,
                agg_method=agg_method,
                group_type=group_type,
                group_threshold=group_threshold,
                mix_type=mix_type,
                enable_paths=enable_paths,
                output_dim=output_dim
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"DMRGCN+GPGraph Model: {d_in}->{d_h}->{output_dim}, T_pred={T_pred}")
        logger.info(f"Encoder: {'MultiModal' if use_multimodal else 'Standard'}")
        logger.info(f"Head: {'Simple' if use_simple_head else 'GPGraph'}")

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    @checked_shape
    def forward(self,
                X: torch.Tensor,                        # [B, T_obs, N, d_in]
                A_obs: torch.Tensor,                    # [B, T_obs, N, N] or [B, N, N]
                M_obs: torch.Tensor,                    # [B, T_obs, N]
                A_pred: Optional[torch.Tensor] = None,  # [B, T_pred, N, N] (optional)
                M_pred: Optional[torch.Tensor] = None,  # [B, T_pred, N] (optional)
                return_intermediate: bool = False,
                return_groups: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass through unified model
        
        Args:
            X: Input features [B, T_obs, N, d_in]
            A_obs: Observed adjacency [B, T_obs, N, N] or [B, N, N]
            M_obs: Observation mask [B, T_obs, N]
            A_pred: Prediction adjacency [B, T_pred, N, N] (optional)
            M_pred: Prediction mask [B, T_pred, N] (optional)
            return_intermediate: Whether to return intermediate results
            return_groups: Whether to return group assignments
            
        Returns:
            ΔY: Predicted deltas [B, T_pred, N, 2]
            Additional returns based on flags:
            - intermediate: Dict with encoder outputs, etc.
            - groups: Group assignments [B, N]
        """
        # Validate and log input shapes
        assert_shape(X, 'b t n d', 'input_features')
        assert_shape(M_obs, 'b t n', 'observation_mask')
        
        B, T_obs, N, d_in = X.shape
        assert d_in == self.d_in, f"Input dimension mismatch: got {d_in}, expected {self.d_in}"
        
        # Ensure adjacency has time axis
        A_obs = ensure_time_axis(A_obs, T_obs)
        assert_shape(A_obs, 'b t n n', 'observed_adjacency')
        
        # Create prediction mask if not provided
        if M_pred is None:
            # Use last observation mask as template
            M_last = M_obs[:, -1:, :]  # [B, 1, N]
            M_pred = M_last.expand(-1, self.T_pred, -1)  # [B, T_pred, N]
        else:
            assert_shape(M_pred, 'b t n', 'prediction_mask')
            assert M_pred.shape[1] == self.T_pred, f"Prediction length mismatch"
        
        log_shape("model_input", X=X, A_obs=A_obs, M_obs=M_obs, M_pred=M_pred)
        
        # Validate model I/O
        input_dict = {'X': X, 'A_obs': A_obs, 'M_obs': M_obs, 'M_pred': M_pred}
        
        # Encoder forward pass
        if self.use_multimodal:
            # Split input into position and velocity (if d_in >= 4)
            if d_in >= 4:
                X_pos = X[..., :2]
                X_vel = X[..., 2:4]
                H = self.encoder(X_pos, A_obs, M_obs, X_vel)
            else:
                # Use positions only, compute velocities internally
                H = self.encoder(X, A_obs, M_obs)
        else:
            H = self.encoder(X, A_obs, M_obs)
        
        # Validate encoder output
        assert_shape(H, 'b t n d', 'encoded_features')
        assert H.shape[-1] == self.d_h, f"Encoder output dimension mismatch"
        
        log_shape("after_encoder", H=H)
        
        # Head forward pass
        if self.use_simple_head:
            delta_Y = self.head(H, M_obs, M_pred)
            group_indices = None
        else:
            if return_groups:
                delta_Y, group_indices = self.head(
                    H, A_obs, M_obs, M_pred, A_pred, return_groups=True
                )
            else:
                delta_Y = self.head(H, A_obs, M_obs, M_pred, A_pred, return_groups=False)
                group_indices = None
        
        # Validate output
        assert_shape(delta_Y, 'b t n d', 'predicted_deltas')
        assert delta_Y.shape[1] == self.T_pred, f"Output time dimension mismatch"
        assert delta_Y.shape[-1] == self.output_dim, f"Output feature dimension mismatch"
        
        # Final I/O validation
        output_dict = {'delta_Y': delta_Y}
        validate_model_io(input_dict, output_dict)
        
        log_shape("model_output", delta_Y=delta_Y)
        
        # Prepare return values
        returns = [delta_Y]
        
        if return_intermediate:
            intermediate = {
                'encoded_features': H,
                'encoder_type': 'multimodal' if self.use_multimodal else 'standard',
                'head_type': 'simple' if self.use_simple_head else 'gpgraph'
            }
            returns.append(intermediate)
        
        if return_groups and group_indices is not None:
            returns.append(group_indices)
        
        return returns[0] if len(returns) == 1 else tuple(returns)

    def compute_loss(self,
                     delta_Y_pred: torch.Tensor,     # [B, T_pred, N, 2]
                     delta_Y_true: torch.Tensor,     # [B, T_pred, N, 2]
                     M_pred: torch.Tensor,           # [B, T_pred, N]
                     loss_type: str = 'mse') -> torch.Tensor:
        """Compute prediction loss
        
        Args:
            delta_Y_pred: Predicted deltas [B, T_pred, N, 2]
            delta_Y_true: Ground truth deltas [B, T_pred, N, 2]
            M_pred: Prediction mask [B, T_pred, N]
            loss_type: Loss function type ('mse', 'l1', 'huber')
            
        Returns:
            loss: Scalar loss value
        """
        # Validate shapes
        assert_shape(delta_Y_pred, 'b t n d', 'predicted_deltas')
        assert_shape(delta_Y_true, 'b t n d', 'true_deltas')
        assert_shape(M_pred, 'b t n', 'prediction_mask')
        
        # Apply mask to both predictions and targets
        delta_Y_pred_masked = mask_features(delta_Y_pred, M_pred)
        delta_Y_true_masked = mask_features(delta_Y_true, M_pred)
        
        # Compute loss
        if loss_type == 'mse':
            loss_per_element = (delta_Y_pred_masked - delta_Y_true_masked) ** 2
        elif loss_type == 'l1':
            loss_per_element = torch.abs(delta_Y_pred_masked - delta_Y_true_masked)
        elif loss_type == 'huber':
            loss_per_element = torch.nn.functional.huber_loss(
                delta_Y_pred_masked, delta_Y_true_masked, reduction='none'
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Sum across coordinate dimensions
        loss_per_element = loss_per_element.sum(dim=-1)  # [B, T_pred, N]
        
        # Masked average
        valid_elements = M_pred.sum()
        if valid_elements > 0:
            loss = (loss_per_element * M_pred).sum() / valid_elements
        else:
            loss = torch.tensor(0.0, device=delta_Y_pred.device)
        
        return loss

    def predict_trajectories(self,
                           X: torch.Tensor,              # [B, T_obs, N, d_in]
                           A_obs: torch.Tensor,          # [B, T_obs, N, N]
                           M_obs: torch.Tensor,          # [B, T_obs, N]
                           M_pred: torch.Tensor,         # [B, T_pred, N]
                           return_absolute: bool = True
                           ) -> torch.Tensor:            # [B, T_pred, N, 2]
        """Generate trajectory predictions
        
        Args:
            X: Input features [B, T_obs, N, d_in]
            A_obs: Observed adjacency [B, T_obs, N, N]
            M_obs: Observation mask [B, T_obs, N]
            M_pred: Prediction mask [B, T_pred, N]
            return_absolute: Whether to return absolute positions
            
        Returns:
            trajectories: Predicted trajectories [B, T_pred, N, 2]
        """
        with torch.no_grad():
            # Get delta predictions
            delta_Y = self.forward(X, A_obs, M_obs, M_pred=M_pred)
            
            if return_absolute:
                # Convert deltas to absolute positions
                # Assume X contains positions in first 2 dimensions
                last_pos = X[:, -1, :, :2]  # [B, N, 2]
                
                # Cumulative sum of deltas starting from last observed position
                positions = torch.zeros_like(delta_Y)
                positions[:, 0] = last_pos + delta_Y[:, 0]
                
                for t in range(1, self.T_pred):
                    positions[:, t] = positions[:, t-1] + delta_Y[:, t]
                
                return positions
            else:
                return delta_Y

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info"""
        return {
            'model_type': 'DMRGCN+GPGraph',
            'input_dim': self.d_in,
            'hidden_dim': self.d_h,
            'gpgraph_dim': self.d_gp_in,
            'prediction_length': self.T_pred,
            'output_dim': self.output_dim,
            'encoder_type': 'multimodal' if self.use_multimodal else 'standard',
            'head_type': 'simple' if self.use_simple_head else 'gpgraph',
            'enabled_paths': self.enable_paths,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
