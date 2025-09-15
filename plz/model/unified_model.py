import torch
import torch.nn as nn
import torch.nn.functional as F
from .enhanced_dmrgcn import EnhancedDMRGCN
from .group_aware_predictor import GroupAwarePredictor, generate_identity_matrix


class BaselineTrajectoryModel(nn.Module):
    """Simplified baseline model compatible with group predictor"""
    def __init__(self, in_dims=3, out_dims=5, hidden_dims=64, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(in_dims, hidden_dims)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims, hidden_dims) for _ in range(3)
        ])
        self.output_projection = nn.Linear(hidden_dims, out_dims)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.PReLU()
        
    def forward(self, graph, identity, mask=None):
        """
        Args:
            graph: Input graph data [batch, time, num_ped, features]
            identity: Identity matrices (not used in this simplified version)
            mask: Optional mask for group interactions
        """
        # Reshape for processing
        batch_size, time_steps, num_ped, features = graph.shape
        x = graph.reshape(-1, features)  # [batch*time*ped, features]
        
        # Apply transformations
        x = self.input_projection(x)
        x = self.activation(x)
        
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
        
        x = self.output_projection(x)
        
        # Reshape back
        x = x.reshape(batch_size, time_steps, num_ped, -1)
        
        # Apply mask if provided (for intra-group interactions)
        if mask is not None:
            # Apply group mask to suppress inter-group interactions
            mask_expanded = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, ped, ped]
            # Simple masking approach - can be enhanced
            x = x * mask_expanded.mean(dim=-1, keepdim=True)
        
        return x.permute(0, 3, 1, 2)  # [batch, features, time, ped]


class UnifiedTrajectoryPredictor(nn.Module):
    """Unified model combining Enhanced DMRGCN with Group-aware prediction"""
    
    def __init__(self, n_stgcn=1, n_tpcnn=4, input_feat=2, output_feat=5, 
                 seq_len=8, pred_seq_len=12, kernel_size=3,
                 d_type='velocity_aware', d_th='learned', mix_type='attention',
                 group_type=(True, True, True), weight_share=True):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        self.output_feat = output_feat
        
        # Enhanced DMRGCN backbone
        self.dmrgcn_backbone = EnhancedDMRGCN(
            n_stgcn=n_stgcn, n_tpcnn=n_tpcnn, input_feat=input_feat,
            output_feat=64, seq_len=seq_len, pred_seq_len=pred_seq_len,
            kernel_size=kernel_size
        )
        
        # Baseline model for group predictor
        self.baseline_model = BaselineTrajectoryModel(
            in_dims=3, out_dims=output_feat, hidden_dims=64
        )
        
        # Group-aware predictor
        self.group_predictor = GroupAwarePredictor(
            baseline_model=self.baseline_model,
            in_channels=64,  # Output from DMRGCN backbone
            out_channels=output_feat,
            obs_seq_len=seq_len,
            pred_seq_len=pred_seq_len,
            d_type=d_type,
            d_th=d_th,
            mix_type=mix_type,
            group_type=group_type,
            weight_share=weight_share
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(64 + output_feat, output_feat, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(output_feat, output_feat, kernel_size=1)
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Conv2d(output_feat, output_feat * 2, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(output_feat * 2, output_feat, kernel_size=1)
        )
        
    def forward(self, v_rel, a_matrices):
        """
        Args:
            v_rel: Relative trajectory data [batch, 2, time, num_ped]
            a_matrices: Adjacency matrices [batch, relation, time, num_ped, num_ped]
        Returns:
            prediction: Predicted trajectories
            indices: Group indices
            auxiliary_info: Dictionary with auxiliary information
        """
        batch_size, _, time_steps, num_ped = v_rel.shape
        
        # Step 1: Enhanced DMRGCN processing
        dmrgcn_features, enhanced_adjacency = self.dmrgcn_backbone(v_rel, a_matrices)
        
        # Step 2: Prepare absolute trajectories for group predictor
        # Convert relative to absolute (simple cumulative sum for demo)
        v_abs = torch.cumsum(v_rel, dim=2)
        
        # Step 3: Group-aware prediction
        group_features, group_indices, velocity = self.group_predictor(v_abs, dmrgcn_features)
        
        # Step 4: Feature fusion
        fused_features = torch.cat([dmrgcn_features, group_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # Step 5: Final prediction
        prediction = self.prediction_head(fused_features)
        
        # Auxiliary information
        auxiliary_info = {
            'dmrgcn_features': dmrgcn_features,
            'group_features': group_features,
            'enhanced_adjacency': enhanced_adjacency,
            'velocity': velocity,
            'group_indices': group_indices
        }
        
        return prediction, group_indices, auxiliary_info
    
    def compute_loss(self, prediction, target, group_indices=None, aux_info=None):
        """
        Compute comprehensive loss including group-aware components
        """
        # Basic prediction loss
        prediction_loss = F.mse_loss(prediction, target)
        
        # Group consistency loss
        group_loss = 0.0
        if group_indices is not None and aux_info is not None:
            group_loss = self._compute_group_consistency_loss(
                prediction, target, group_indices, aux_info
            )
        
        # Velocity consistency loss
        velocity_loss = 0.0
        if aux_info is not None and 'velocity' in aux_info:
            velocity_loss = self._compute_velocity_consistency_loss(
                prediction, aux_info['velocity']
            )
        
        total_loss = prediction_loss + 0.1 * group_loss + 0.05 * velocity_loss
        
        return {
            'total_loss': total_loss,
            'prediction_loss': prediction_loss,
            'group_loss': group_loss,
            'velocity_loss': velocity_loss
        }
    
    def _compute_group_consistency_loss(self, prediction, target, group_indices, aux_info):
        """Compute group consistency loss to ensure coherent group behavior"""
        if group_indices is None:
            return torch.tensor(0.0, device=prediction.device)
        
        group_loss = 0.0
        unique_groups = group_indices.unique()
        
        for group_id in unique_groups:
            group_mask = (group_indices == group_id)
            if group_mask.sum() > 1:  # Only for groups with multiple members
                # Get predictions for group members
                group_pred = prediction[:, :, :, group_mask]
                group_target = target[:, :, :, group_mask]
                
                # Compute pairwise consistency within group
                n_members = group_pred.size(-1)
                for i in range(n_members):
                    for j in range(i + 1, n_members):
                        # Encourage similar motion patterns within groups
                        pred_diff = group_pred[:, :, :, i] - group_pred[:, :, :, j]
                        target_diff = group_target[:, :, :, i] - group_target[:, :, :, j]
                        group_loss += F.mse_loss(pred_diff, target_diff)
        
        return group_loss
    
    def _compute_velocity_consistency_loss(self, prediction, velocity):
        """Compute velocity consistency loss"""
        # Compute predicted velocity
        if prediction.size(2) > 1:
            pred_velocity = prediction[:, :2, 1:, :] - prediction[:, :2, :-1, :]
            target_velocity = velocity[:, :, 1:, :]
            return F.mse_loss(pred_velocity, target_velocity)
        else:
            return torch.tensor(0.0, device=prediction.device)
    
    def sample_trajectories(self, v_rel, a_matrices, n_samples=20, use_group_sampling=True):
        """
        Sample multiple trajectory predictions
        """
        prediction, group_indices, aux_info = self.forward(v_rel, a_matrices)
        
        # Extract distribution parameters (assuming multivariate Gaussian output)
        mu = prediction[:, :2, :, :]  # Mean
        log_sigma = prediction[:, 2:4, :, :]  # Log variance
        sigma = torch.exp(log_sigma)
        
        # Generate samples
        samples = []
        for _ in range(n_samples):
            if use_group_sampling and group_indices is not None:
                # Group-aware sampling
                noise = self._generate_group_aware_noise(
                    mu.shape, group_indices, sigma
                )
            else:
                # Standard sampling
                noise = torch.randn_like(mu) * sigma
            
            sample = mu + noise
            samples.append(sample)
        
        return torch.stack(samples, dim=0), group_indices
    
    def _generate_group_aware_noise(self, shape, group_indices, sigma):
        """Generate noise that respects group structure"""
        noise = torch.randn(shape, device=sigma.device)
        
        # Apply group correlation
        unique_groups = group_indices.unique()
        for group_id in unique_groups:
            group_mask = (group_indices == group_id)
            if group_mask.sum() > 1:
                # Generate correlated noise for group members
                group_noise = torch.randn(shape[0], shape[1], shape[2], device=sigma.device)
                noise[:, :, :, group_mask] = (
                    0.7 * group_noise.unsqueeze(-1) + 
                    0.3 * noise[:, :, :, group_mask]
                )
        
        return noise * sigma
