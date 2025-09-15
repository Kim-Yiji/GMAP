import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def clip_adjacency_matrix(A, min=-1e10, max=1e10):
    """Returns the clipped Adjacency matrix with min and max values."""
    A_c = A.clamp(min=min, max=max)
    A_c[A_c == min] = 0
    A_c[A_c == max] = 0
    A_c[A_c > 0] = 1
    return A_c


def get_disentangled_adjacency_matrix(A, split=[]):
    """Returns the tensor of clipped Adjacency matrix split by list values."""
    if len(split) == 0:
        return A.unsqueeze(1)  # Add dimension to match expected output

    split.sort()
    split = split + [1e10]

    A_d = []
    for i in range(len(split) - 1):
        A_d.append(clip_adjacency_matrix(A, min=split[i], max=split[i + 1]))

    return torch.stack(A_d, dim=1)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
         - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak


def normalized_laplacian_tilde_matrix(A, eps=1e-8):
    """Calculate normalized Laplacian matrix"""
    # Add self-loop
    A_tilde = A + torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    
    # Calculate degree matrix
    D = torch.diag_embed(A_tilde.sum(dim=-1))
    
    # Calculate D^(-1/2)
    D_sqrt_inv = torch.pow(D + eps, -0.5)
    D_sqrt_inv[torch.isinf(D_sqrt_inv)] = 0
    
    # Calculate normalized Laplacian
    L_norm = torch.eye(A.size(-1), device=A.device, dtype=A.dtype) - \
             D_sqrt_inv @ A_tilde @ D_sqrt_inv
    
    return L_norm


def drop_edge(A, drop_rate, training=True):
    """Drop edges randomly during training"""
    if not training or drop_rate <= 0:
        return A
    
    mask = torch.rand_like(A) > drop_rate
    return A * mask


class ConvTemporalGraphical(nn.Module):
    """Temporal Graph Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, 
                 t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 
                              kernel_size=(t_kernel_size, 1), padding=(t_padding, 0),
                              stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == x.size(0)
        assert A.size(-1) == self.kernel_size
        
        x = self.conv(x)
        x = x.view(x.size(0), self.kernel_size, -1, x.size(-2), x.size(-1))
        x = torch.einsum('nkctv,nkwv->nctw', x, normalized_laplacian_tilde_matrix(
            drop_edge(A, 0.8, self.training)))
        
        return x.contiguous(), A


class MultiRelationalGCN(nn.Module):
    """Enhanced Multi-Relational GCN with velocity and acceleration awareness"""
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, 
                 t_padding=0, t_dilation=1, bias=True, relation=3):  # 3 relations: position, velocity, acceleration
        super().__init__()
        self.kernel_size = kernel_size
        self.relation = relation
        self.out_channels = out_channels
        
        # Separate convolutions for different relations
        # Single convolution layer for all relations (following original DMRGCN)
        self.conv = nn.Conv2d(in_channels, out_channels * relation, kernel_size=(t_kernel_size, 1),
                             padding=(t_padding, 0), stride=(t_stride, 1), 
                             dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        # A should have shape [batch, relation, time, vertices, vertices]
        # x should have shape [batch, channels, time, vertices]
        # Following original DMRGCN convention
        assert A.size(0) == x.size(0)  # batch
        assert A.size(1) == self.relation  # relations
        assert A.size(2) == self.kernel_size  # time dimension should match kernel_size

        # Apply convolution (following original DMRGCN)
        x = self.conv(x)
        
        # Reshape to separate relations
        x = x.view(x.size(0), self.relation, self.out_channels, x.size(-2), x.size(-1))
        
        # Apply graph convolution using original DMRGCN approach
        # A: [batch, relation, time, vertices, vertices]
        # x: [batch, relation, out_channels, time, vertices]
        x_out = torch.einsum('nrtwv,nrctv->nctw', 
                           normalized_laplacian_tilde_matrix(drop_edge(A, 0.8, self.training)), 
                           x)

        return x_out.contiguous(), A


class EnhancedSTDMRGCN(nn.Module):
    """Enhanced Spatio-Temporal DMRGCN with velocity and acceleration"""
    def __init__(self, in_channels, out_channels, kernel_size, use_mdn=False, stride=1, 
                 dropout=0, residual=True, split=[], relation=3):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn
        self.split = split
        self.relation = relation

        # Enhanced GCNs for multiple relations
        self.gcns = nn.ModuleList()
        for r in range(self.relation):
            # For empty split lists, use 1 relation; otherwise use split length
            rel_count = max(1, len(split[r])) if r < len(split) else 2
            self.gcns.append(MultiRelationalGCN(in_channels, out_channels, kernel_size[1], 
                                                relation=rel_count))

        # Temporal convolution
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
        res = self.residual(x)

        # Split adjacency matrices by relation
        A_r = torch.split(A, 1, dim=1)
        
        for r in range(self.relation):
            if r < len(self.split):
                A_ = get_disentangled_adjacency_matrix(A_r[r].squeeze(dim=1), self.split[r])
            else:
                A_ = A_r[r].squeeze(dim=1)
            
            x_a, _ = self.gcns[r](x, A_)

            if r == 0:
                x_r = x_a
            else:
                x_r = x_r + x_a

        x = self.tcn(x_r) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A


class VelocityAccelerationExtractor(nn.Module):
    """Extract velocity and acceleration from trajectory data"""
    def __init__(self):
        super().__init__()

    def forward(self, traj_rel):
        """
        Args:
            traj_rel: Relative trajectory data [batch, 2, time, num_ped]
        Returns:
            velocity, acceleration
        """
        # Velocity is already relative displacement
        velocity = traj_rel
        
        # Acceleration is the difference between consecutive velocities
        if traj_rel.size(2) > 1:
            acceleration = torch.zeros_like(velocity)
            acceleration[:, :, 1:, :] = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]
        else:
            acceleration = torch.zeros_like(velocity)
        
        return velocity, acceleration


class EnhancedDMRGCN(nn.Module):
    """Enhanced DMRGCN with velocity, acceleration, and improved spatial reasoning"""
    def __init__(self, n_stgcn=1, n_tpcnn=4, input_feat=2, output_feat=5, 
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super().__init__()
        
        self.n_stgcn = n_stgcn
        self.n_tpcnn = n_tpcnn
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        
        # Velocity and acceleration extractor
        self.vel_acc_extractor = VelocityAccelerationExtractor()
        
        # Enhanced STGCN layers
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(EnhancedSTDMRGCN(input_feat, 64, [kernel_size, seq_len], 
                                             split=[[], [1.0], [1.5]], relation=3))
        
        for j in range(1, self.n_stgcn):
            self.st_gcns.append(EnhancedSTDMRGCN(64, 64, [kernel_size, seq_len], 
                                                 split=[[], [1.0], [1.5]], relation=3))

        # Channel projection from 64 to seq_len for TPCNN compatibility
        self.channel_projection = nn.Conv2d(64, seq_len, 1)
        
        # Temporal prediction CNN (following original DMRGCN)
        self.tpcnns = nn.ModuleList()
        # First layer: from seq_len to pred_seq_len
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, kernel_size))
        
        for j in range(1, self.n_tpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size))
            
        # Output layer for enhanced features
        self.output_layer = nn.Conv2d(pred_seq_len, output_feat, 1)
        
    def forward(self, v, a):
        """
        Args:
            v: Input velocity data [batch, 2, time, num_ped]
            a: Input adjacency matrices [batch, relation, time, num_ped, num_ped]
        """
        # Extract velocity and acceleration
        velocity, acceleration = self.vel_acc_extractor(v)
        
        # Enhance adjacency matrices with velocity and acceleration information
        a_enhanced = self._enhance_adjacency_matrices(a, velocity, acceleration)
        
        # Apply STGCN layers
        for k in range(self.n_stgcn):
            v, a_enhanced = self.st_gcns[k](v, a_enhanced)

        # Project 64 channels back to seq_len for TPCNN
        # v shape: [batch, 64, time, vertices] -> [batch, seq_len, time, vertices]
        v = self.channel_projection(v)

        # NCTV -> NTCV (following original DMRGCN)
        v = v.permute(0, 2, 1, 3)
        
        # Apply temporal prediction layers  
        for k in range(self.n_tpcnn):
            v = F.prelu(self.tpcnns[k](v))

        # NTCV -> NCTV (following original DMRGCN)
        v = v.permute(0, 2, 1, 3)

        # Generate output
        v = self.output_layer(v)
        
        return v, a_enhanced
    
    def _enhance_adjacency_matrices(self, a, velocity, acceleration):
        """Enhance adjacency matrices with velocity and acceleration information"""
        batch_size, _, time_steps, num_ped = velocity.shape
        
        # Create enhanced adjacency matrices
        a_enhanced = torch.zeros(batch_size, 3, time_steps, num_ped, num_ped, 
                                 device=a.device, dtype=a.dtype)
        
        # Original position-based adjacency (from input)
        if a.size(1) >= 1:
            a_enhanced[:, 0] = a[:, 0]  # First adjacency matrix
        if a.size(1) >= 2:
            a_enhanced[:, 1] = a[:, 1]  # Second adjacency matrix if available
        
        # If we have a third channel, use velocity-based adjacency
        if a.size(1) < 3:
            # Velocity-based adjacency
            for t in range(time_steps):
                vel_t = velocity[:, :, t, :]  # [batch, 2, num_ped]
                for i in range(num_ped):
                    for j in range(num_ped):
                        if i != j:
                            # Velocity similarity
                            vel_diff = torch.norm(vel_t[:, :, i] - vel_t[:, :, j], p=2, dim=1)
                            a_enhanced[:, 2, t, i, j] = torch.exp(-vel_diff)
        else:
            a_enhanced[:, 2] = a[:, 2]  # Use existing third adjacency
        
        return a_enhanced
