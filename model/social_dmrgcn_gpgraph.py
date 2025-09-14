"""
통합 DMRGCN + GP-Graph 모델
DMRGCN의 Multi-Relational Graph Convolution과 GP-Graph의 Group-based Processing을 결합

이 파일은 두 개의 최신 보행자 궤적 예측 모델을 통합한 핵심 구현체입니다:

1. DMRGCN (AAAI 2021): Multi-Relational Graph Convolution 기반
   - 원본 출처: https://github.com/InhwanBae/DMRGCN
   - 특징: Disentangled Multi-scale Aggregation, Global Temporal Aggregation

2. GP-Graph (ECCV 2022): Group-based Processing 기반  
   - 원본 출처: https://github.com/InhwanBae/GPGraph
   - 특징: Pedestrian Group Pooling/Unpooling, Group Hierarchy Graph

통합 방식:
- DMRGCN을 베이스 모델로 사용
- GP-Graph의 Group 기능을 모듈로 추가
- Local Density와 Group Size 특징 추가 (미팅에서 요청된 기능)
"""

import torch
import torch.nn as nn
from .dmrgcn import st_dmrgcn, social_dmrgcn  # DMRGCN 원본 모델들
from .gpgraph_modules import GroupGenerator, GroupIntegrator, DensityGroupFeatureExtractor, generate_adjacency_matrix  # GP-Graph 모듈들


class SocialDMRGCN_GPGraph(nn.Module):
    """
    DMRGCN과 GP-Graph를 통합한 모델
    
    통합 구조:
    1. DMRGCN을 베이스 궤적 예측기로 사용
    2. GP-Graph의 그룹 기반 처리 통합
    3. Local Density와 Group Size 특징 추가
    
    이 모델은 미팅에서 요청된 다음 기능들을 모두 포함합니다:
    - 속도, 가속도, 상대변위 (DMRGCN에서 제공)
    - 그룹 표현 (GP-Graph에서 제공)
    - 밀도와 그룹 크기 (새로 추가된 특징)
    """
    
    def __init__(self, n_stgcn=1, n_tpcnn=4, output_feat=5, kernel_size=3, 
                 seq_len=8, pred_seq_len=12, split=[], relation=2,
                 # GP-Graph parameters
                 group_d_type='learned_l2norm', group_d_th='learned', 
                 group_mix_type='mlp', group_type=(True, True, True),
                 # Density/Group size parameters
                 density_radius=2.0, group_size_threshold=2,
                 # Feature integration
                 use_density=True, use_group_size=True, use_group_processing=True):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        self.use_density = use_density
        self.use_group_size = use_group_size
        self.use_group_processing = use_group_processing
        
        # Base DMRGCN model
        self.dmrgcn = social_dmrgcn(
            n_stgcn=n_stgcn, n_tpcnn=n_tpcnn, output_feat=output_feat,
            kernel_size=kernel_size, seq_len=seq_len, pred_seq_len=pred_seq_len,
            split=split, relation=relation
        )
        
        # GP-Graph modules
        if self.use_group_processing:
            self.group_gen = GroupGenerator(
                d_type=group_d_type, th=group_d_th, 
                in_channels=2, hid_channels=8, n_head=1
            )
            self.group_mix = GroupIntegrator(
                mix_type=group_mix_type, n_mix=sum(group_type),
                out_channels=output_feat, pred_seq_len=pred_seq_len
            )
            self.group_type = group_type
        
        # Density and Group Size Feature Extractor
        if self.use_density or self.use_group_size:
            self.feature_extractor = DensityGroupFeatureExtractor(
                density_radius=density_radius,
                group_size_threshold=group_size_threshold
            )
        
        # Feature integration layers
        self.feature_integration = self._build_feature_integration(
            output_feat, use_density, use_group_size, use_group_processing
        )
    
    def _build_feature_integration(self, output_feat, use_density, use_group_size, use_group_processing):
        """Build feature integration layers based on enabled features"""
        input_dim = output_feat
        
        if use_density:
            input_dim += 1  # density feature
        if use_group_size:
            input_dim += 1  # group size feature
        if use_group_processing:
            input_dim += output_feat  # group processing output
            
        if input_dim > output_feat:
            return nn.Sequential(
                nn.Conv2d(input_dim, output_feat, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(output_feat, output_feat, kernel_size=1)
            )
        else:
            return nn.Identity()
    
    def forward(self, v_obs, A_obs):
        """
        Forward pass of the integrated model
        Args:
            v_obs: (batch, 2, seq_len, num_ped) - observed trajectories
            A_obs: (batch, relation, seq_len, num_ped, num_ped) - adjacency matrices
        Returns:
            v_pred: (batch, output_feat, pred_seq_len, num_ped) - predicted trajectories
            group_indices: (num_ped,) - group assignments (if group processing enabled)
        """
        batch, _, seq_len, num_ped = v_obs.shape
        
        # Get absolute positions for feature extraction
        v_abs = v_obs.permute(0, 2, 3, 1)  # (batch, seq_len, num_ped, 2)
        
        # Extract additional features
        additional_features = []
        group_indices = None
        
        if self.use_density or self.use_group_size:
            features = self.feature_extractor(v_abs, group_indices)
            
            if self.use_density:
                density = features['density']  # (batch, seq_len, num_ped)
                # Expand to match prediction length
                density_pred = density[:, -1:].repeat(1, self.pred_seq_len, 1)  # (batch, pred_seq_len, num_ped)
                additional_features.append(density_pred.permute(0, 2, 1).unsqueeze(1))  # (batch, 1, pred_seq_len, num_ped)
            
            if self.use_group_size:
                group_size = features['group_size']  # (num_ped,)
                # Expand to match prediction format
                group_size_pred = group_size.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, num_ped)
                group_size_pred = group_size_pred.repeat(batch, 1, self.pred_seq_len, 1)  # (batch, 1, pred_seq_len, num_ped)
                additional_features.append(group_size_pred)
        
        # Base DMRGCN prediction
        v_pred_base, _ = self.dmrgcn(v_obs, A_obs)
        
        if self.use_group_processing:
            # Group-based processing
            v_stack = []
            
            # Original individual-level processing
            if self.group_type[0]:
                v_orig = v_pred_base
                v_stack.append(v_orig)
            
            # Group generation
            v_rel = v_obs.permute(0, 2, 3, 1)  # (batch, seq_len, num_ped, 2)
            v_rel, group_indices = self.group_gen(v_rel, v_abs, hard=True)
            
            # Inter-group processing
            if self.group_type[1]:
                v_e = self.group_gen.ped_group_pool(v_rel, group_indices)
                v_e = v_e.permute(0, 3, 1, 2)  # (batch, 2, seq_len, num_groups)
                
                # Create adjacency matrix for groups
                A_e = generate_adjacency_matrix(v_e)
                v_e_pred, _ = self.dmrgcn(v_e, A_e.unsqueeze(0).unsqueeze(0))
                
                # Unpool back to individual level
                v_e_pred = v_e_pred.permute(0, 2, 3, 1)  # (batch, seq_len, num_groups, 2)
                v_e_pred = self.group_gen.ped_group_unpool(v_e_pred, group_indices)
                v_e_pred = v_e_pred.permute(0, 3, 1, 2)  # (batch, 2, seq_len, num_ped)
                v_stack.append(v_e_pred)
            
            # Intra-group processing
            if self.group_type[2]:
                v_i = v_rel.permute(0, 3, 1, 2)  # (batch, 2, seq_len, num_ped)
                mask = self.group_gen.ped_group_mask(group_indices)
                A_i = generate_adjacency_matrix(v_i) * mask
                v_i_pred, _ = self.dmrgcn(v_i, A_i.unsqueeze(0).unsqueeze(0))
                v_stack.append(v_i_pred)
            
            # Group integration
            v_pred_group = self.group_mix(v_stack)
            additional_features.append(v_pred_group)
        
        # Integrate all features
        if additional_features:
            # Concatenate additional features
            all_features = [v_pred_base] + additional_features
            v_pred = torch.cat(all_features, dim=1)
            
            # Apply feature integration
            v_pred = self.feature_integration(v_pred)
        else:
            v_pred = v_pred_base
        
        return v_pred, group_indices


def create_integrated_model(n_stgcn=1, n_tpcnn=4, output_feat=5, kernel_size=3, 
                           seq_len=8, pred_seq_len=12, split=[], relation=2,
                           group_d_type='learned_l2norm', group_d_th='learned',
                           group_mix_type='mlp', group_type=(True, True, True),
                           density_radius=2.0, group_size_threshold=2,
                           use_density=True, use_group_size=True, use_group_processing=True):
    """
    Factory function to create the integrated model
    """
    return SocialDMRGCN_GPGraph(
        n_stgcn=n_stgcn, n_tpcnn=n_tpcnn, output_feat=output_feat,
        kernel_size=kernel_size, seq_len=seq_len, pred_seq_len=pred_seq_len,
        split=split, relation=relation, group_d_type=group_d_type,
        group_d_th=group_d_th, group_mix_type=group_mix_type,
        group_type=group_type, density_radius=density_radius,
        group_size_threshold=group_size_threshold, use_density=use_density,
        use_group_size=use_group_size, use_group_processing=use_group_processing
    )
