# Shape utilities for enforcing unified tensor dimensions
# All external interfaces must follow [B, T, N, d] convention

import torch
import logging
from typing import Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def assert_shape(tensor: torch.Tensor, spec: str, name: str = "tensor") -> None:
    """Assert tensor shape matches specification
    
    Args:
        tensor: Input tensor
        spec: Shape specification like 'b t n d' or 'b t n n' 
        name: Tensor name for error messages
        
    Raises:
        AssertionError: If shape doesn't match specification
        
    Examples:
        >>> x = torch.randn(2, 8, 5, 64)
        >>> assert_shape(x, 'b t n d', 'features')  # OK
        >>> assert_shape(x, 'b t n n', 'adjacency')  # AssertionError
    """
    spec_parts = spec.strip().lower().split()
    actual_shape = list(tensor.shape)
    
    if len(actual_shape) != len(spec_parts):
        raise AssertionError(
            f"{name} has {len(actual_shape)} dimensions {actual_shape}, "
            f"but spec '{spec}' expects {len(spec_parts)} dimensions"
        )
    
    # Check dimension constraints
    dim_values = {}
    for i, (actual, expected) in enumerate(zip(actual_shape, spec_parts)):
        if expected in dim_values:
            if dim_values[expected] != actual:
                raise AssertionError(
                    f"{name} dimension {i} ({expected}) = {actual}, "
                    f"but previous {expected} = {dim_values[expected]}"
                )
        else:
            dim_values[expected] = actual
    
    logger.debug(f"✓ {name} shape {actual_shape} matches '{spec}'")


def log_shape(tag: str, **named_tensors) -> None:
    """Log shapes of named tensors for debugging
    
    Args:
        tag: Description tag
        **named_tensors: Named tensors to log
        
    Example:
        >>> log_shape("after_encoder", X=features, A=adjacency, M=mask)
    """
    shapes = {name: list(tensor.shape) for name, tensor in named_tensors.items()}
    logger.info(f"🔍 {tag}: {shapes}")


def ensure_time_axis(A: torch.Tensor, T: int) -> torch.Tensor:
    """Ensure adjacency matrix has time axis
    
    Args:
        A: Adjacency matrix [B, N, N] or [B, T, N, N]
        T: Expected time dimension
        
    Returns:
        A_expanded: [B, T, N, N]
    """
    if A.dim() == 3:
        # [B, N, N] -> [B, T, N, N]
        A_expanded = A.unsqueeze(1).expand(-1, T, -1, -1)
        logger.debug(f"Expanded adjacency from {list(A.shape)} to {list(A_expanded.shape)}")
        return A_expanded
    elif A.dim() == 4:
        # Already [B, T, N, N]
        assert A.shape[1] == T, f"Time dimension mismatch: got {A.shape[1]}, expected {T}"
        return A
    else:
        raise ValueError(f"Invalid adjacency shape: {A.shape}, expected [B, N, N] or [B, T, N, N]")


def ensure_relation_axis(A: torch.Tensor, R_expected: Optional[int] = None) -> torch.Tensor:
    """Ensure adjacency matrix has relation axis
    
    Args:
        A: Adjacency matrix [B, T, N, N] or [B, T, R, N, N]
        R_expected: Expected number of relations (if None, add single relation)
        
    Returns:
        A_with_relations: [B, T, R, N, N]
    """
    if A.dim() == 4:
        # [B, T, N, N] -> [B, T, 1, N, N]
        A_rel = A.unsqueeze(2)
        if R_expected and R_expected > 1:
            A_rel = A_rel.expand(-1, -1, R_expected, -1, -1)
        logger.debug(f"Added relation axis: {list(A.shape)} -> {list(A_rel.shape)}")
        return A_rel
    elif A.dim() == 5:
        # Already [B, T, R, N, N]
        if R_expected:
            assert A.shape[2] == R_expected, f"Relation mismatch: got {A.shape[2]}, expected {R_expected}"
        return A
    else:
        raise ValueError(f"Invalid adjacency shape: {A.shape}")


def mask_features(H: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply mask to features
    
    Args:
        H: Features [B, T, N, d]
        M: Mask [B, T, N]
        
    Returns:
        H_masked: Masked features [B, T, N, d]
    """
    assert_shape(H, 'b t n d', 'features')
    assert_shape(M, 'b t n', 'mask')
    
    # Broadcast mask to feature dimensions
    H_masked = H * M.unsqueeze(-1)  # [B, T, N, d] * [B, T, N, 1]
    
    log_shape("mask_features", H=H, M=M, H_masked=H_masked)
    return H_masked


def mask_adj(A: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply mask to adjacency matrix
    
    Args:
        A: Adjacency [B, T, N, N] or [B, T, R, N, N]
        M: Mask [B, T, N]
        
    Returns:
        A_masked: Masked adjacency with same shape as A
    """
    if A.dim() == 4:
        assert_shape(A, 'b t n n', 'adjacency')
    elif A.dim() == 5:
        assert_shape(A, 'b t r n n', 'multi_relation_adjacency')
    else:
        raise ValueError(f"Invalid adjacency shape: {A.shape}")
    
    assert_shape(M, 'b t n', 'mask')
    
    # Create masks for both source and target nodes
    M_i = M.unsqueeze(-1)      # [B, T, N, 1] - source nodes
    M_j = M.unsqueeze(-2)      # [B, T, 1, N] - target nodes
    
    if A.dim() == 5:
        # Multi-relation case: [B, T, R, N, N]
        M_i = M_i.unsqueeze(2)  # [B, T, 1, N, 1]
        M_j = M_j.unsqueeze(2)  # [B, T, 1, 1, N]
    
    # Apply mask: edge exists only if both nodes are valid
    A_masked = A * M_i * M_j
    
    log_shape("mask_adj", A=A, M=M, A_masked=A_masked)
    return A_masked


def aggregate_time_features(H: torch.Tensor, M: torch.Tensor, method: str = 'last') -> torch.Tensor:
    """Aggregate features across time dimension
    
    Args:
        H: Features [B, T, N, d]
        M: Mask [B, T, N]
        method: Aggregation method ('last', 'mean', 'max')
        
    Returns:
        H_agg: Aggregated features [B, N, d]
    """
    assert_shape(H, 'b t n d', 'features')
    assert_shape(M, 'b t n', 'mask')
    
    B, T, N, d = H.shape
    
    if method == 'last':
        # Find last valid timestep for each agent
        valid_lengths = M.sum(dim=1)  # [B, N] - number of valid timesteps per agent
        last_indices = (valid_lengths - 1).clamp(min=0).long()  # [B, N]
        
        # Gather last valid features
        batch_indices = torch.arange(B, device=H.device).unsqueeze(1).expand(-1, N)  # [B, N]
        agent_indices = torch.arange(N, device=H.device).unsqueeze(0).expand(B, -1)  # [B, N]
        
        H_agg = H[batch_indices, last_indices, agent_indices, :]  # [B, N, d]
        
    elif method == 'mean':
        # Masked mean across time
        H_masked = mask_features(H, M)  # Apply mask first
        valid_counts = M.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, N]
        H_agg = H_masked.sum(dim=1) / valid_counts.transpose(-1, -2)  # [B, N, d]
        
    elif method == 'max':
        # Masked max across time
        H_masked = mask_features(H, M)
        H_masked = H_masked.masked_fill(~M.unsqueeze(-1), float('-inf'))
        H_agg = H_masked.max(dim=1)[0]  # [B, N, d]
        
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    log_shape("aggregate_time", H=H, M=M, H_agg=H_agg)
    return H_agg


def pad_sequence_batch(sequences: list, max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to uniform length
    
    Args:
        sequences: List of tensors with shape [T_i, N, d]
        max_length: Maximum length (if None, use max in batch)
        
    Returns:
        padded: [B, T_max, N, d]
        mask: [B, T_max, N] - True for valid positions
    """
    if not sequences:
        raise ValueError("Empty sequence list")
    
    # Get dimensions
    sample_seq = sequences[0]
    N, d = sample_seq.shape[-2:]
    B = len(sequences)
    
    if max_length is None:
        max_length = max(seq.shape[0] for seq in sequences)
    
    # Create padded tensor and mask
    device = sample_seq.device
    dtype = sample_seq.dtype
    
    padded = torch.zeros(B, max_length, N, d, device=device, dtype=dtype)
    mask = torch.zeros(B, max_length, N, device=device, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        T_i = seq.shape[0]
        length = min(T_i, max_length)
        
        padded[i, :length] = seq[:length]
        mask[i, :length] = True
    
    log_shape("pad_sequence_batch", padded=padded, mask=mask)
    return padded, mask


def validate_model_io(input_dict: Dict[str, torch.Tensor], 
                     output_dict: Dict[str, torch.Tensor]) -> None:
    """Validate model input/output shapes match unified convention
    
    Args:
        input_dict: Input tensors with expected shapes
        output_dict: Output tensors with expected shapes
    """
    # Validate inputs
    if 'X' in input_dict:
        assert_shape(input_dict['X'], 'b t n d', 'input_features')
    
    if 'A_obs' in input_dict:
        A = input_dict['A_obs']
        if A.dim() == 4:
            assert_shape(A, 'b t n n', 'observed_adjacency')
        elif A.dim() == 5:
            assert_shape(A, 'b t r n n', 'multi_relation_adjacency')
    
    if 'M_obs' in input_dict:
        assert_shape(input_dict['M_obs'], 'b t n', 'observed_mask')
    
    if 'M_pred' in input_dict:
        assert_shape(input_dict['M_pred'], 'b t n', 'prediction_mask')
    
    # Validate outputs
    if 'delta_Y' in output_dict:
        assert_shape(output_dict['delta_Y'], 'b t n d', 'predicted_deltas')
        # Check that last dimension is 2 for (x, y) coordinates
        assert output_dict['delta_Y'].shape[-1] == 2, "Delta predictions must have shape [..., 2]"
    
    logger.info("✅ Model I/O shapes validated successfully")


# Global shape validation toggle for debugging
ENABLE_SHAPE_VALIDATION = True

def set_shape_validation(enabled: bool) -> None:
    """Enable/disable shape validation globally"""
    global ENABLE_SHAPE_VALIDATION
    ENABLE_SHAPE_VALIDATION = enabled
    logger.info(f"Shape validation {'enabled' if enabled else 'disabled'}")


def checked_shape(func):
    """Decorator to add automatic shape checking to functions"""
    def wrapper(*args, **kwargs):
        if ENABLE_SHAPE_VALIDATION:
            result = func(*args, **kwargs)
            # Log function call with tensor shapes
            tensor_args = {f"arg_{i}": arg for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)}
            tensor_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
            if tensor_args or tensor_kwargs:
                log_shape(f"{func.__name__}", **tensor_args, **tensor_kwargs)
            return result
        else:
            return func(*args, **kwargs)
    return wrapper
