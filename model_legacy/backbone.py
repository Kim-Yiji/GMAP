# Modular DMRGCN backbone for shared processing across multiple graph types
# Based on DMRGCN architecture with enhancements for group-aware modeling

import torch
import torch.nn as nn
from .utils import (
    normalized_laplacian_tilde_matrix, 
    drop_edge, 
    get_disentangled_adjacency_matrix,
    clip_adjacency_matrix
)


class ConvTemporalGraphical(nn.Module):
    """Basic module for applying a graph convolution.
    
    Based on ST-GCN implementation with improvements for batch processing.
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(t_kernel_size, 1), 
            padding=(t_padding, 0),
            stride=(t_stride, 1), 
            dilation=(t_dilation, 1), 
            bias=bias
        )

    def forward(self, x, A):
        """
        Args:
            x: (B, C, T, N) - input features
            A: (B, K, T, N, N) - adjacency matrices
        Returns:
            x: (B, C, T, N) - output features
            A: (B, K, T, N, N) - adjacency matrices (unchanged)
        """
        # Batch calculation for A matrix
        assert A.size(0) == x.size(0)
        assert A.size(1) == self.kernel_size
        
        x = self.conv(x)
        
        # Apply graph convolution with normalized Laplacian
        # A: (B, K, T, N, N) -> need to process each (B, T, N, N) separately
        B, K, T, N, _ = A.shape
        x_out = torch.zeros_like(x)
        
        for k in range(K):
            A_k = A[:, k, :, :, :]  # (B, T, N, N)
            # Process each time step
            for t in range(T):
                A_kt = A_k[:, t, :, :]  # (B, N, N)
                # Apply dropout and normalization
                A_kt_norm = normalized_laplacian_tilde_matrix(
                    drop_edge(A_kt, 0.8, self.training)
                )
                # Graph convolution: (B, C, N) @ (B, N, N) -> (B, C, N)
                x_out[:, :, t, :] += torch.bmm(x[:, :, t, :], A_kt_norm)
        
        return x_out.contiguous(), A


class MultiRelationalGCN(nn.Module):
    """Multi-relational graph convolution for handling multiple relation types"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True, relation=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.relation = relation
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(
            in_channels, out_channels * relation, 
            kernel_size=(t_kernel_size, 1), 
            padding=(t_padding, 0),
            stride=(t_stride, 1), 
            dilation=(t_dilation, 1), 
            bias=bias
        )

    def forward(self, x, A):
        """
        Args:
            x: (B, C, T, N) - input features
            A: (B, R, T, N, N) - multi-relational adjacency matrices
        Returns:
            x: (B, C_out, T, N) - output features
            A: (B, R, T, N, N) - adjacency matrices (unchanged)
        """
        assert A.size(0) == x.size(0)
        assert A.size(1) == self.relation
        # kernel_size refers to spatial dimension, which should match the node dimension
        assert A.size(4) == x.size(3), f"Spatial dimension mismatch: A.size(4)={A.size(4)}, x.size(3)={x.size(3)}"
        
        x = self.conv(x)  # (B, C_out * R, T, N)
        B, _, T, N = x.shape
        
        # Reshape to separate relations
        x = x.view(B, self.relation, self.out_channels, T, N)
        
        # Apply graph convolution for each relation
        x_out = torch.zeros(B, self.out_channels, T, N, device=x.device, dtype=x.dtype)
        
        for r in range(self.relation):
            A_r = A[:, r, :, :, :]  # (B, T, N, N)
            x_r = x[:, r, :, :, :]   # (B, C_out, T, N)
            
            for t in range(T):
                A_rt = A_r[:, t, :, :]  # (B, N, N)
                A_rt_norm = normalized_laplacian_tilde_matrix(
                    drop_edge(A_rt, 0.8, self.training)
                )
                x_out[:, :, t, :] += torch.bmm(x_r[:, :, t, :], A_rt_norm)
        
        return x_out.contiguous(), A


class DMRGCNBlock(nn.Module):
    """Single DMRGCN block with disentangled multi-relational processing"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 use_mdn=False, stride=1, dropout=0, residual=True,
                 split=[], relation=2):
        super().__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.split = split
        self.relation = relation
        
        # Spatial Edge Processing
        self.gcns = nn.ModuleList()
        for r in range(self.relation):
            split_r = split[r] if isinstance(split, list) and len(split) > r else []
            self.gcns.append(
                MultiRelationalGCN(
                    in_channels, out_channels, kernel_size[1], 
                    relation=len(split_r) + 1 if split_r else 1
                )
            )
        
        # Temporal Edge Processing
        self.tcn = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.Dropout(dropout, inplace=True),
        )
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            )
        
        self.prelu = nn.PReLU()

    def forward(self, x, A):
        """
        Args:
            x: (B, C, T, N) - input features
            A: (B, R, T, N, N) - multi-relational adjacency matrices
        Returns:
            x: (B, C_out, T, N) - output features
            A: (B, R, T, N, N) - adjacency matrices (unchanged)
        """
        res = self.residual(x)
        
        # Process each relation type
        A_relations = torch.split(A, 1, dim=1)  # List of (B, 1, T, N, N)
        
        x_combined = None
        for r in range(self.relation):
            A_r = A_relations[r].squeeze(1)  # (B, T, N, N)
            
            # Apply disentanglement if specified
            if self.split and len(self.split) > r and self.split[r]:
                A_r_dis = get_disentangled_adjacency_matrix(A_r, self.split[r])  # (B, R_sub, T, N, N)
            else:
                A_r_dis = A_r.unsqueeze(1)  # (B, 1, T, N, N)
            
            x_r, _ = self.gcns[r](x, A_r_dis)
            
            if x_combined is None:
                x_combined = x_r
            else:
                x_combined = x_combined + x_r
        
        # Temporal convolution and residual
        x = self.tcn(x_combined) + res
        
        if not self.use_mdn:
            x = self.prelu(x)
        
        return x, A


class DMRGCNBackbone(nn.Module):
    """Modular DMRGCN backbone for shared processing across different graph types
    
    This backbone can be shared across agent-level, intra-group, and inter-group processing.
    """
    
    def __init__(self, input_channels=2, hidden_channels=[64, 64, 64, 64, 5], 
                 kernel_size=(3, 1), stride=1, dropout=0.1, 
                 split_configs=None, relation=2, use_mdn=False):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.use_mdn = use_mdn
        
        # Default split configurations for each relation
        if split_configs is None:
            split_configs = [[] for _ in range(relation)]
        self.split_configs = split_configs
        
        # Build DMRGCN layers
        self.dmrgcn_layers = nn.ModuleList()
        
        channels = [input_channels] + hidden_channels
        for i in range(self.num_layers):
            is_last_layer = (i == self.num_layers - 1)
            
            layer = DMRGCNBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_size,
                use_mdn=use_mdn and is_last_layer,
                stride=stride,
                dropout=dropout if not is_last_layer else 0,
                residual=True,
                split=split_configs,
                relation=relation
            )
            self.dmrgcn_layers.append(layer)

    def forward(self, x, A):
        """
        Args:
            x: (B, C, T, N) - input features (e.g., relative positions)
            A: (B, R, T, N, N) - multi-relational adjacency matrices
        Returns:
            x: (B, C_out, T, N) - output features
        """
        for layer in self.dmrgcn_layers:
            x, A = layer(x, A)
        
        return x


class PredictionHead(nn.Module):
    """Prediction head for trajectory forecasting with Gaussian parameters"""
    
    def __init__(self, input_channels, pred_len=12, output_dim=5):
        """
        Args:
            input_channels: number of input feature channels
            pred_len: prediction sequence length
            output_dim: 5 for Gaussian (mu_x, mu_y, sig_x, sig_y, rho)
        """
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        
        self.predictor = nn.Sequential(
            nn.Conv2d(input_channels, output_dim, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=(3, 1), padding=(1, 0))
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T, N) - backbone features
        Returns:
            pred: (B, output_dim, pred_len, N) - prediction parameters
        """
        # Apply prediction head
        pred = self.predictor(x)  # (B, output_dim, T, N)
        
        # Take the last pred_len time steps
        pred = pred[:, :, -self.pred_len:, :]
        
        return pred
