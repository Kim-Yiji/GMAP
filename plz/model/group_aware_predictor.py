import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class GroupGenerator(nn.Module):
    """Enhanced Group Generator for trajectory prediction"""
    def __init__(self, d_type='learned', th=1., in_channels=16, hid_channels=32, n_head=1, dropout=0):
        super().__init__()
        self.d_type = d_type
        
        if d_type == 'learned':
            self.group_cnn = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, 1),
                nn.ReLU(),
                nn.BatchNorm2d(hid_channels),
                nn.Dropout(dropout, inplace=True),
                nn.Conv2d(hid_channels, n_head, 1),
            )
        elif d_type == 'estimate_th':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, n_head, 1),)
        elif d_type == 'learned_l2norm':
            self.group_cnn = nn.Sequential(
                nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 1), padding=(1, 0))
            )
        elif d_type == 'velocity_aware':
            # New: Velocity-aware grouping
            self.group_cnn = nn.Sequential(
                nn.Conv2d(in_channels + 2, hid_channels, 1),  # +2 for velocity
                nn.ReLU(),
                nn.BatchNorm2d(hid_channels),
                nn.Dropout(dropout, inplace=True),
                nn.Conv2d(hid_channels, n_head, 1),
            )
            
        self.th = th if type(th) == float else nn.Parameter(torch.Tensor([1]))

    def find_group_indices(self, v, dist_mat):
        n_ped = v.size(-1)
        mask = torch.ones_like(dist_mat).mul(1e4).triu()
        top_row, top_column = torch.nonzero(
            dist_mat.tril(diagonal=-1).add(mask).le(self.th), as_tuple=True
        )
        indices_raw = torch.arange(n_ped, dtype=top_row.dtype, device=v.device)
        
        for r, c in zip(top_row, top_column):
            mask = indices_raw == indices_raw[r]
            indices_raw[mask] = c
            
        indices_uniq = indices_raw.unique()
        indices_map = torch.arange(indices_uniq.size(0), dtype=top_row.dtype, device=v.device)
        indices = torch.zeros_like(indices_raw)
        
        for i, j in zip(indices_uniq, indices_map):
            indices[indices_raw == i] = j
            
        return indices

    def group_backprop_trick_threshold(self, v, dist_mat, tau=1, hard=False):
        """Enhanced grouping with hard/soft assignment"""
        sig = (-(dist_mat - self.th) / tau).sigmoid()
        sig_norm = sig / sig.sum(dim=0, keepdim=True)
        v_soft = v @ sig_norm
        return (v - v_soft).detach() + v_soft if hard else v_soft

    def forward(self, v, v_abs, velocity=None, tau=0.1, hard=True):
        assert v.size(0) == 1
        n_ped = v.size(-1)

        # Measure similarity between pedestrian pairs
        if self.d_type == 'euclidean':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned_l2norm':
            temp = self.group_cnn(v_abs).unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-1, -2)).reshape(temp.size(0), -1, n_ped, n_ped)
            temp = self.group_cnn(temp).exp()
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1)
        elif self.d_type == 'velocity_aware' and velocity is not None:
            # Enhanced: Consider both position and velocity for grouping
            v_combined = torch.cat([v_abs, velocity], dim=1)
            temp = v_combined.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-1, -2)).reshape(temp.size(0), -1, n_ped, n_ped)
            temp = self.group_cnn(temp).exp()
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1)
        else:
            # Default to euclidean
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)

        dist_mat = dist_mat.squeeze(dim=0).mean(dim=0)
        indices = self.find_group_indices(v, dist_mat)
        v = self.group_backprop_trick_threshold(v, dist_mat, tau=tau, hard=hard)
        return v, indices

    @staticmethod
    def ped_group_pool(v, indices):
        """Pool pedestrians into groups"""
        assert v.size(-1) == indices.size(0)
        n_ped = v.size(-1)
        n_ped_pool = indices.unique().size(0)
        v_pool = torch.zeros(v.shape[:-1] + (n_ped_pool,), device=v.device)
        v_pool.index_add_(-1, indices, v)
        v_pool_num = torch.zeros((v.size(0), 1, 1, n_ped_pool), device=v.device)
        v_pool_num.index_add_(-1, indices, torch.ones((v.size(0), 1, 1, n_ped), device=v.device))
        v_pool /= v_pool_num
        return v_pool

    @staticmethod
    def ped_group_unpool(v, indices):
        """Unpool group features back to individuals"""
        assert v.size(-1) == indices.unique().size(0)
        return torch.index_select(input=v, dim=-1, index=indices)

    @staticmethod
    def ped_group_mask(indices):
        """Create mask for intra-group interactions"""
        mask = torch.eye(indices.size(0), dtype=torch.bool, device=indices.device)
        for i in indices.unique():
            idx_list = torch.nonzero(indices.eq(i), as_tuple=False).squeeze()
            if idx_list.numel() > 1:
                for idx in idx_list:
                    mask[idx, idx_list] = 1
        return mask


class GroupIntegrator(nn.Module):
    """Enhanced Group Integrator for combining different interaction types"""
    def __init__(self, mix_type='mean', n_mix=3, out_channels=5, pred_seq_len=12):
        super().__init__()
        self.mix_type = mix_type
        self.pred_seq_len = pred_seq_len
        
        if mix_type == 'mlp':
            self.st_gcns_mix = nn.Sequential(
                nn.PReLU(),
                nn.Conv2d(out_channels * pred_seq_len * n_mix, out_channels * pred_seq_len,
                          kernel_size=1),
            )
        elif mix_type == 'cnn':
            self.st_gcns_mix = nn.Sequential(
                nn.PReLU(),
                nn.Conv2d(out_channels * n_mix, out_channels,
                          kernel_size=(3, 1), padding=(1, 0))
            )
        elif mix_type == 'attention':
            # New: Attention-based integration
            self.attention_weights = nn.Parameter(torch.ones(n_mix) / n_mix)
            self.attention_mlp = nn.Sequential(
                nn.Linear(out_channels * n_mix, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, n_mix),
                nn.Softmax(dim=-1)
            )

    def forward(self, v_stack):
        n_batch, n_ped = v_stack[0].shape[0], v_stack[0].shape[3]
        
        if self.mix_type == 'sum':
            v = torch.stack(v_stack, dim=0).sum(dim=0)
        elif self.mix_type == 'mean':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
        elif self.mix_type == 'mlp':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v_stack_cat = torch.cat(v_stack, dim=1).reshape(n_batch, -1, 1, n_ped)
            v = v + self.st_gcns_mix(v_stack_cat).view(n_batch, -1, self.pred_seq_len, n_ped)
        elif self.mix_type == 'cnn':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v = v + self.st_gcns_mix(torch.cat(v_stack, dim=1))
        elif self.mix_type == 'attention':
            # Attention-based weighted combination
            v_tensor = torch.stack(v_stack, dim=0)  # [n_mix, batch, channels, time, ped]
            
            # Calculate attention weights
            v_global = v_tensor.mean(dim=[2, 3, 4])  # [n_mix, batch, channels]
            v_global_flat = v_global.permute(1, 0, 2).reshape(n_batch, -1)  # [batch, n_mix*channels]
            attention_weights = self.attention_mlp(v_global_flat)  # [batch, n_mix]
            
            # Apply attention weights
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            v = (v_tensor * attention_weights.unsqueeze(2)).sum(dim=0)
        else:
            raise NotImplementedError
            
        return v


def generate_identity_matrix(v):
    """Generate spatial and temporal identity matrix"""
    i = [
        torch.eye(v.size(3), device=v.device).repeat(v.size(2), 1, 1),
        torch.eye(v.size(2), device=v.device).repeat(v.size(3), 1, 1)
    ]
    return i


class EnhancedSelfAttention(nn.Module):
    """Enhanced Self-Attention with velocity awareness"""
    def __init__(self, in_dims=2, d_model=64, num_heads=4, velocity_aware=True):
        super().__init__()
        
        input_dims = in_dims + 2 if velocity_aware else in_dims  # +2 for velocity
        self.velocity_aware = velocity_aware
        
        self.embedding = nn.Linear(input_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.scaled_factor = torch.sqrt(torch.Tensor([d_model]))
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads

    def forward(self, x, velocity=None, mask=None):
        """
        Args:
            x: [batch_size, seq_len, in_dims]
            velocity: [batch_size, seq_len, 2] (optional)
            mask: attention mask (optional)
        """
        if self.velocity_aware and velocity is not None:
            x = torch.cat([x, velocity], dim=-1)
            
        embeddings = self.embedding(x)
        query = self.query(embeddings)
        key = self.key(embeddings)
        value = self.value(embeddings)
        
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention / self.scaled_factor.to(attention.device)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
            
        attention = self.softmax(attention)
        output = torch.matmul(attention, value)
        
        return output, attention


class GroupAwarePredictor(nn.Module):
    """Group-aware trajectory predictor combining DMRGCN with GPGraph concepts"""
    def __init__(self, baseline_model, in_channels=2, out_channels=5, obs_seq_len=8, pred_seq_len=12,
                 d_type='velocity_aware', d_th='learned', mix_type='attention', 
                 group_type=None, weight_share=True):
        super().__init__()

        self.baseline_model = baseline_model
        self.obs_seq_len = obs_seq_len
        self.pred_seq_len = pred_seq_len
        self.mix_type = mix_type
        self.weight_share = weight_share

        group_type = (True,) * 3 if group_type is None else group_type
        self.include_original = group_type[0]
        self.include_inter_group = group_type[1]
        self.include_intra_group = group_type[2]

        # Enhanced group generator with velocity awareness
        self.group_gen = GroupGenerator(d_type=d_type, th=d_th, in_channels=in_channels, 
                                        hid_channels=16)
        
        # Enhanced group integrator
        self.group_mix = GroupIntegrator(mix_type=mix_type, n_mix=sum(group_type),
                                         out_channels=out_channels, pred_seq_len=pred_seq_len)
        
        # Velocity extractor
        self.velocity_extractor = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, v_abs, v_rel):
        """
        Args:
            v_abs: Absolute trajectory data [batch, channels, time, num_ped]
            v_rel: Relative trajectory data [batch, channels, time, num_ped]
        """
        v_stack = []
        
        # Extract velocity features
        velocity = self.velocity_extractor(v_rel)
        
        # Pedestrian graph (original individual interactions)
        if self.include_original:
            v = v_rel
            i = generate_identity_matrix(v)
            v = v.permute(0, 2, 3, 1)
            
            if self.weight_share:
                v = self.baseline_model(v, i)
            else:
                v = self.baseline_model[0](v, i)
                
            v = v.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v)

        # Enhanced group-based interactions
        v_rel_grouped, indices = self.group_gen(v_rel, v_abs, velocity, hard=True)

        if self.include_inter_group:
            # Inter-group interaction (between different groups)
            v_e = self.group_gen.ped_group_pool(v_rel_grouped, indices)
            i_e = generate_identity_matrix(v_e)
            v_e = v_e.permute(0, 2, 3, 1)
            
            if self.weight_share:
                v_e = self.baseline_model(v_e, i_e)
            else:
                v_e = self.baseline_model[1](v_e, i_e)
                
            v_e = v_e.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_e = self.group_gen.ped_group_unpool(v_e, indices)
            v_stack.append(v_e)

        if self.include_intra_group:
            # Intra-group interaction (within the same group)
            v_i = v_rel_grouped
            mask = self.group_gen.ped_group_mask(indices)
            i_i = generate_identity_matrix(v_i)
            v_i = v_i.permute(0, 2, 3, 1)
            
            if self.weight_share:
                v_i = self.baseline_model(v_i, i_i, mask)
            else:
                v_i = self.baseline_model[2](v_i, i_i, mask)
                
            v_i = v_i.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v_i)

        # Enhanced group integration
        v = self.group_mix(v_stack)

        return v, indices, velocity


class AdaptiveGroupSampler:
    """Adaptive sampling strategy for group-aware trajectory prediction"""
    def __init__(self, stack_n=1000, fast_sample=True):
        self.stack_n = stack_n
        self.pre_samples = []
        self.fast_sample = fast_sample

    def randn(self, n, k, d, group_indices=None):
        """Generate adaptive random samples based on group structure"""
        if group_indices is not None:
            # Group-aware sampling
            unique_groups = group_indices.unique()
            n_groups = len(unique_groups)
            
            randn_sample = []
            for _ in range(n):
                group_samples = []
                for group_id in unique_groups:
                    group_mask = group_indices == group_id
                    group_size = group_mask.sum().item()
                    
                    if group_size > 1:
                        # Generate correlated samples for group members
                        group_center = torch.randn(d) * 0.5
                        group_noise = torch.randn(group_size, d) * 0.3
                        group_sample = group_center.unsqueeze(0) + group_noise
                    else:
                        # Single pedestrian group
                        group_sample = torch.randn(1, d)
                    
                    group_samples.append(group_sample)
                
                # Combine all group samples
                full_sample = torch.zeros(len(group_indices), d)
                start_idx = 0
                for i, group_id in enumerate(unique_groups):
                    group_mask = group_indices == group_id
                    group_size = group_mask.sum().item()
                    full_sample[group_mask] = group_samples[i]
                
                randn_sample.append(full_sample.unsqueeze(0))
            
            return torch.cat(randn_sample, dim=0).numpy()
        else:
            # Standard sampling
            return np.random.randn(n, k, d)


# Global adaptive sampler instance
adaptive_random = AdaptiveGroupSampler()
