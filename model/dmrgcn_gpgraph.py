# Main integrated model: DMRGCN + GP-Graph
# Combines DMRGCN backbone with group-aware processing for trajectory prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

from .backbone import DMRGCNBackbone, PredictionHead
from .gpgraph_adapter import (
    GroupAssignment, GroupIntegration, 
    ped_group_pool, ped_group_unpool, ped_group_mask
)
from .utils import get_disentangled_adjacency_matrix


class DMRGCNGPGraph(nn.Module):
    """Integrated DMRGCN + GP-Graph model for group-aware trajectory prediction
    
    Forward path:
    1. Group assignment based on last observed positions
    2. Build three graph types: agent, intra-group, inter-group  
    3. Process each graph type with shared DMRGCN backbone
    4. Fuse features and predict trajectories with Gaussian parameters
    """
    
    def __init__(self, 
                 obs_len=8, 
                 pred_len=12,
                 input_dim=2,
                 hidden_dims=[64, 64, 64, 64, 5],
                 kernel_size=(3, 1),
                 dropout=0.1,
                 # Group-aware parameters
                 group_type='euclidean',
                 group_threshold=2.0,
                 mix_type='mean',
                 enable_paths={'agent': True, 'intra': True, 'inter': True},
                 # DMRGCN parameters  
                 distance_scales=[0.5, 1.0, 2.0],
                 split_configs=None,
                 share_backbone=True,
                 # Training parameters
                 use_mdn=True,
                 st_estimator=False):
        super().__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1]
        self.share_backbone = share_backbone
        self.enable_paths = enable_paths
        self.distance_scales = distance_scales
        self.use_mdn = use_mdn
        
        # Count enabled paths
        self.n_paths = sum(enable_paths.values())
        if self.n_paths == 0:
            raise ValueError("At least one processing path must be enabled")
        
        # Group assignment module
        self.group_assignment = GroupAssignment(
            d_type=group_type,
            th=group_threshold, 
            in_channels=input_dim,
            hid_channels=32,
            st_estimator=st_estimator
        )
        
        # DMRGCN backbone(s)
        if share_backbone:
            # Single shared backbone
            self.backbone = DMRGCNBackbone(
                input_channels=input_dim,
                hidden_channels=hidden_dims,
                kernel_size=kernel_size,
                dropout=dropout,
                split_configs=split_configs,
                relation=len(distance_scales),
                use_mdn=use_mdn
            )
        else:
            # Separate backbones for each path
            self.backbones = nn.ModuleDict()
            for path_name, enabled in enable_paths.items():
                if enabled:
                    self.backbones[path_name] = DMRGCNBackbone(
                        input_channels=input_dim,
                        hidden_channels=hidden_dims,
                        kernel_size=kernel_size,
                        dropout=dropout,
                        split_configs=split_configs,
                        relation=len(distance_scales),
                        use_mdn=use_mdn
                    )
        
        # Feature integration
        self.feature_integration = GroupIntegration(
            mix_type=mix_type,
            n_mix=self.n_paths,
            out_channels=self.output_dim,
            pred_seq_len=pred_len,
            feature_dim=hidden_dims[-2] if len(hidden_dims) > 1 else hidden_dims[0]
        )
        
        # Prediction head (if not using MDN in backbone)
        if not use_mdn:
            self.prediction_head = PredictionHead(
                input_channels=self.output_dim,
                pred_len=pred_len,
                output_dim=5  # Gaussian parameters
            )

    def build_agent_graphs(self, V, A):
        """Build agent-level graphs from adjacency matrices
        
        Args:
            V: (B, T, N, 2) - node features (relative displacements)
            A: (B, R, T, N, N) - multi-relational adjacency matrices
            
        Returns:
            graphs: (B, R, T, N, N) - processed adjacency matrices
        """
        # Apply distance-based filtering using scales
        B, R, T, N, _ = A.shape
        graphs = torch.zeros_like(A)
        
        for r, scale in enumerate(self.distance_scales):
            if r < R:
                # Use distance adjacency (A[:, 1] is distance, A[:, 0] is displacement)
                A_dist = A[:, min(1, R-1), :, :, :]  # (B, T, N, N)
                # Apply scale threshold
                graphs[:, r, :, :, :] = (A_dist <= scale).float() * A_dist
        
        return graphs

    def build_intra_group_graphs(self, A, group_indices):
        """Build intra-group graphs by masking inter-group connections
        
        Args:
            A: (B, R, T, N, N) - agent-level adjacency matrices
            group_indices: (B, N) - group assignments
            
        Returns:
            intra_graphs: (B, R, T, N, N) - intra-group adjacency matrices
        """
        B, R, T, N, _ = A.shape
        
        # Create group mask
        group_mask = ped_group_mask(group_indices)  # (N, N)
        group_mask = group_mask.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, N)
        group_mask = group_mask.expand(B, R, T, -1, -1)
        
        # Apply mask to adjacency matrices
        intra_graphs = A * group_mask
        
        return intra_graphs

    def build_inter_group_graphs(self, V, A, group_indices):
        """Build inter-group graphs through pooling and unpooling
        
        Args:
            V: (B, T, N, 2) - node features
            A: (B, R, T, N, N) - agent-level adjacency matrices  
            group_indices: (B, N) - group assignments
            
        Returns:
            inter_graphs: (B, R, T, N, N) - inter-group adjacency matrices
            group_features: (B, T, Ng, 2) - group-level features
        """
        B, T, N, _ = V.shape
        R = A.shape[1]
        
        # Convert to (B, C, T, N) format for pooling
        V_pool = V.permute(0, 3, 1, 2)  # (B, 2, T, N)
        
        # Pool to group level
        group_features = ped_group_pool(V_pool, group_indices)  # (B, 2, T, Ng)
        Ng = group_features.shape[-1]
        
        # Create inter-group adjacency matrices
        inter_graphs = torch.zeros(B, R, T, Ng, Ng, device=A.device, dtype=A.dtype)
        
        # Build connections between groups based on agent-level connections
        for b in range(B):
            for r in range(R):
                for t in range(T):
                    A_rt = A[b, r, t, :, :]  # (N, N)
                    group_idx = group_indices[b]  # (N,)
                    
                    # Create inter-group connections
                    for i in range(N):
                        for j in range(N):
                            gi = group_idx[i].item()
                            gj = group_idx[j].item()
                            if gi != gj and A_rt[i, j] > 0:
                                inter_graphs[b, r, t, gi, gj] = 1.0
        
        # Convert back to agent level through unpooling
        group_features_agent = ped_group_unpool(group_features, group_indices)  # (B, 2, T, N)
        
        # Create agent-level inter-group adjacency
        inter_adj_agent = torch.zeros(B, R, T, N, N, device=A.device, dtype=A.dtype)
        for b in range(B):
            group_idx = group_indices[b]
            for i in range(N):
                for j in range(N):
                    gi = group_idx[i].item()
                    gj = group_idx[j].item()
                    if gi != gj:
                        # Copy inter-group connection strength
                        inter_adj_agent[b, :, :, i, j] = inter_graphs[b, :, :, gi, gj]
        
        return inter_adj_agent, group_features.permute(0, 2, 3, 1)  # (B, T, Ng, 2)

    def forward(self, V_obs, A_obs, return_intermediate=False):
        """Forward pass through the integrated model
        
        Args:
            V_obs: (B, T, N, 2) - observed node features
            A_obs: (B, R, T, N, N) - observed adjacency matrices
            return_intermediate: whether to return intermediate results
            
        Returns:
            predictions: (B, 5, pred_len, N) - Gaussian parameters
            group_indices: (B, N) - group assignments (if return_intermediate)
            intermediate: dict of intermediate results (if return_intermediate)
        """
        B, T, N, _ = V_obs.shape
        R = A_obs.shape[1]
        
        # Convert to (B, C, T, N) format for processing
        V = V_obs.permute(0, 3, 1, 2)  # (B, 2, T, N)
        A = A_obs  # (B, R, T, N, N)
        
        # 1. Group assignment using last observed features
        V_grouped, group_indices, dist_matrices = self.group_assignment(
            V, V_obs.permute(0, 3, 1, 2), hard=True
        )
        
        # 2. Build different graph types
        feature_list = []
        intermediate_results = {}
        
        # Agent-level processing
        if self.enable_paths['agent']:
            A_agent = self.build_agent_graphs(V_obs, A)
            
            if self.share_backbone:
                feat_agent = self.backbone(V, A_agent)
            else:
                feat_agent = self.backbones['agent'](V, A_agent)
            
            feature_list.append(feat_agent)
            intermediate_results['agent_features'] = feat_agent
            intermediate_results['agent_graphs'] = A_agent
        
        # Intra-group processing  
        if self.enable_paths['intra']:
            A_intra = self.build_intra_group_graphs(A, group_indices)
            
            if self.share_backbone:
                feat_intra = self.backbone(V, A_intra)
            else:
                feat_intra = self.backbones['intra'](V, A_intra)
            
            feature_list.append(feat_intra)
            intermediate_results['intra_features'] = feat_intra
            intermediate_results['intra_graphs'] = A_intra
        
        # Inter-group processing
        if self.enable_paths['inter']:
            A_inter, group_features = self.build_inter_group_graphs(V_obs, A, group_indices)
            
            # For inter-group, we could process group-level features or agent-level with inter-group adjacency
            # Here we use agent-level features with inter-group adjacency
            if self.share_backbone:
                feat_inter = self.backbone(V, A_inter)
            else:
                feat_inter = self.backbones['inter'](V, A_inter)
            
            feature_list.append(feat_inter)
            intermediate_results['inter_features'] = feat_inter
            intermediate_results['inter_graphs'] = A_inter
            intermediate_results['group_features'] = group_features
        
        # 3. Feature integration
        fused_features = self.feature_integration(feature_list)  # (B, C, T, N)
        
        # 4. Prediction
        if self.use_mdn:
            # Backbone already outputs prediction parameters
            predictions = fused_features  # (B, 5, T, N)
            # Take last pred_len timesteps
            predictions = predictions[:, :, -self.pred_len:, :]
        else:
            # Use separate prediction head
            predictions = self.prediction_head(fused_features)  # (B, 5, pred_len, N)
        
        intermediate_results['fused_features'] = fused_features
        intermediate_results['predictions'] = predictions
        
        if return_intermediate:
            return predictions, group_indices, intermediate_results
        else:
            return predictions, group_indices

    def generate_statistics_matrices(self, V):
        """Generate mean and covariance matrices from network output for Gaussian prediction
        
        Args:
            V: (B, 5, T, N) - prediction parameters [mu_x, mu_y, log_sig_x, log_sig_y, rho]
            
        Returns:
            mu: (B, T, N, 2) - predicted means
            cov: (B, T, N, 2, 2) - predicted covariances
        """
        mu = V[:, 0:2, :, :].permute(0, 2, 3, 1)  # (B, T, N, 2)
        
        # Exponential for positive variances
        sx = V[:, 2, :, :].exp()  # (B, T, N)
        sy = V[:, 3, :, :].exp()  # (B, T, N)
        
        # Tanh for correlation in [-1, 1]
        corr = V[:, 4, :, :].tanh()  # (B, T, N)
        
        # Build covariance matrices
        B, T, N = sx.shape
        cov = torch.zeros(B, T, N, 2, 2, device=V.device, dtype=V.dtype)
        
        cov[:, :, :, 0, 0] = sx * sx
        cov[:, :, :, 0, 1] = corr * sx * sy  
        cov[:, :, :, 1, 0] = corr * sx * sy
        cov[:, :, :, 1, 1] = sy * sy
        
        return mu, cov

    def compute_loss(self, predictions, targets, loss_mask=None):
        """Compute negative log-likelihood loss for Gaussian predictions
        
        Args:
            predictions: (B, 5, T, N) - predicted parameters
            targets: (B, T, N, 2) - ground truth trajectories
            loss_mask: (B, T, N) - mask for valid predictions
            
        Returns:
            loss: scalar tensor
        """
        mu, cov = self.generate_statistics_matrices(predictions)  # (B, T, N, 2), (B, T, N, 2, 2)
        
        # Reshape for batch processing
        B, T, N, _ = mu.shape
        mu_flat = mu.reshape(-1, 2)  # (B*T*N, 2)
        cov_flat = cov.reshape(-1, 2, 2)  # (B*T*N, 2, 2)
        targets_flat = targets.reshape(-1, 2)  # (B*T*N, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        eye = torch.eye(2, device=cov.device, dtype=cov.dtype).unsqueeze(0)
        cov_flat = cov_flat + eps * eye
        
        # Compute negative log-likelihood
        try:
            # Use multivariate normal distribution
            diff = targets_flat - mu_flat  # (B*T*N, 2)
            
            # Compute log determinant
            det = torch.det(cov_flat)  # (B*T*N,)
            log_det = torch.log(det + eps)
            
            # Compute quadratic form
            cov_inv = torch.inverse(cov_flat)  # (B*T*N, 2, 2)
            quad_form = torch.bmm(
                diff.unsqueeze(1),  # (B*T*N, 1, 2)
                torch.bmm(cov_inv, diff.unsqueeze(-1))  # (B*T*N, 2, 1)
            ).squeeze()  # (B*T*N,)
            
            # Negative log-likelihood
            nll = 0.5 * (quad_form + log_det + 2 * torch.log(torch.tensor(2 * 3.14159, device=cov.device)))
            
            # Reshape and apply mask
            nll = nll.reshape(B, T, N)
            
            if loss_mask is not None:
                nll = nll * loss_mask
                loss = nll.sum() / loss_mask.sum()
            else:
                loss = nll.mean()
                
        except:
            # Fallback to simpler MSE loss if numerical issues
            loss = F.mse_loss(mu, targets)
        
        return loss

    def predict_trajectories(self, predictions, num_samples=20):
        """Sample trajectories from predicted Gaussian distributions
        
        Args:
            predictions: (B, 5, T, N) - predicted parameters
            num_samples: number of trajectory samples
            
        Returns:
            samples: (num_samples, B, T, N, 2) - sampled trajectories
        """
        mu, cov = self.generate_statistics_matrices(predictions)
        B, T, N, _ = mu.shape
        
        samples = torch.zeros(num_samples, B, T, N, 2, device=predictions.device)
        
        for s in range(num_samples):
            for b in range(B):
                for t in range(T):
                    for n in range(N):
                        # Sample from multivariate normal
                        try:
                            dist = torch.distributions.MultivariateNormal(
                                mu[b, t, n], cov[b, t, n]
                            )
                            samples[s, b, t, n] = dist.sample()
                        except:
                            # Fallback to mean if sampling fails
                            samples[s, b, t, n] = mu[b, t, n]
        
        return samples
