"""
GP-Graph 모듈들을 DMRGCN과 통합하기 위한 파일
원본 GP-Graph (ECCV 2022)에서 가져온 컴포넌트들을 모듈화하여 DMRGCN에 쉽게 통합할 수 있도록 함

원본 출처: https://github.com/InhwanBae/GPGraph
논문: "Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction" (ECCV 2022)
"""

import torch
import torch.nn as nn


class GroupGenerator(nn.Module):
    """
    GP-Graph에서 가져온 그룹 생성기 모듈
    각 보행자를 가장 가능성 높은 행동 그룹에 할당하는 역할
    
    원본 GP-Graph의 핵심 컴포넌트로, 보행자들 간의 유사도를 계산하여
    그룹을 자동으로 형성하는 비지도 학습 방식 사용
    """
    def __init__(self, d_type='learned', th=1., in_channels=16, hid_channels=32, n_head=1, dropout=0):
        """
        그룹 생성기 초기화
        
        Args:
            d_type: 거리 계산 방식 ('learned', 'learned_l2norm', 'euclidean', 'estimate_th')
            th: 그룹 형성을 위한 거리 임계값
            in_channels: 입력 채널 수 (보행자 특징 차원)
            hid_channels: 은닉층 채널 수
            n_head: 출력 헤드 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.d_type = d_type
        
        # 거리 계산 방식에 따른 CNN 네트워크 구성
        if d_type == 'learned':
            # 학습 가능한 거리 계산: 1x1 Conv + ReLU + BatchNorm + Dropout + 1x1 Conv
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, hid_channels, 1),
                                           nn.ReLU(),
                                           nn.BatchNorm2d(hid_channels),
                                           nn.Dropout(dropout, inplace=True),
                                           nn.Conv2d(hid_channels, n_head, 1),)
        elif d_type == 'estimate_th':
            # 임계값 추정 방식: 단순한 1x1 Conv
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, n_head, 1),)
        elif d_type == 'learned_l2norm':
            # L2 정규화된 학습 방식: 3x1 Conv (시간 차원 고려)
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 1), padding=(1, 0)))
        
        # 그룹 형성 임계값 설정 (고정값 또는 학습 가능한 파라미터)
        self.th = th if type(th) == float else nn.Parameter(torch.Tensor([1]))

    def find_group_indices(self, v, dist_mat):
        n_ped = v.size(-1)
        mask = torch.ones_like(dist_mat).mul(1e4).triu()
        top_row, top_column = torch.nonzero(dist_mat.tril(diagonal=-1).add(mask).le(self.th), as_tuple=True)
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

    def find_group_indices_ratio(self, v, dist_mat):
        n_ped = v.size(-1)
        group_num = n_ped - (n_ped + self.th - 1) // self.th
        top_list = (1. / dist_mat).tril(diagonal=-1).view(-1).topk(k=group_num)[1]
        top_row, top_column = top_list // n_ped, top_list % n_ped
        indices_raw = torch.arange(n_ped, dtype=top_list.dtype, device=v.device)
        for r, c in zip(top_row, top_column):
            mask = indices_raw == indices_raw[r]
            indices_raw[mask] = c
        indices_uniq = indices_raw.unique()
        indices_map = torch.arange(indices_uniq.size(0), dtype=top_list.dtype, device=v.device)
        indices = torch.zeros_like(indices_raw)
        for i, j in zip(indices_uniq, indices_map):
            indices[indices_raw == i] = j
        return indices

    def group_backprop_trick_threshold(self, v, dist_mat, tau=1, hard=False):
        """
        The main trick for hard is to do (v_hard - v_soft).detach() + v_soft
        Sample hard categorical using "Straight-through" trick
        """
        sig = (-(dist_mat - self.th) / tau).sigmoid()
        sig_norm = sig / sig.sum(dim=0, keepdim=True)
        v_soft = v @ sig_norm
        return (v - v_soft).detach() + v_soft if hard else v_soft

    def _process_single(self, v, v_abs, tau=0.1, hard=True):
        n_ped = v.size(0)

        # Measure similarity between pedestrian pairs
        if self.d_type == 'euclidean':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned_l2norm':
            temp = self.group_cnn(v_abs.unsqueeze(0)).squeeze(0).unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-1, -2)).reshape(1, -1, n_ped, n_ped)
            temp = self.group_cnn(temp).exp()
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1).squeeze(0)  # symmetric
        elif self.d_type == 'estimate_th':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-2, -1))
            dist_mat = temp.norm(p=2, dim=1)
            self.th = self.group_cnn(temp.reshape(1, -1, n_ped, n_ped)).mean().exp()
        else:
            raise NotImplementedError

        dist_mat = dist_mat.mean(dim=0)
        indices = self.find_group_indices(v.unsqueeze(0), dist_mat)
        v = self.group_backprop_trick_threshold(v.unsqueeze(0), dist_mat.unsqueeze(0), tau=tau, hard=hard).squeeze(0)
        return v, indices

    def forward(self, v, v_abs, tau=0.1, hard=True):
        """
        v: (batch, 2, seq_len, num_ped)
        v_abs: (batch, 2, seq_len, num_ped)
        """
        assert v.dim() == 4 and v_abs.dim() == 4
        assert v.size(0) == 1, "GroupGenerator currently assumes batch size 1."
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
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1)  # symmetric
        elif self.d_type == 'estimate_th':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-2, -1))
            dist_mat = temp.norm(p=2, dim=1)
            self.th = self.group_cnn(temp.reshape(1, -1, n_ped, n_ped)).mean().exp()
        else:
            raise NotImplementedError

        # Aggregate over time and batch (batch is 1)
        dist_mat = dist_mat.squeeze(dim=0).mean(dim=0)  # (num_ped, num_ped)
        indices = self.find_group_indices(v.squeeze(0), dist_mat)
        v = self.group_backprop_trick_threshold(v.squeeze(0), dist_mat.unsqueeze(0), tau=tau, hard=hard).unsqueeze(0)
        return v, indices

    @staticmethod
    def ped_group_pool(v, indices):
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
        assert v.size(-1) == indices.unique().size(0)
        return torch.index_select(input=v, dim=-1, index=indices)

    @staticmethod
    def ped_group_mask(indices):
        mask = torch.eye(indices.size(0), dtype=torch.bool, device=indices.device)
        for i in indices.unique():
            idx_list = torch.nonzero(indices.eq(i))
            for idx in idx_list:
                mask[idx, idx_list] = 1
        return mask


class GroupIntegrator(nn.Module):
    """
    Group Integrator Module from GP-Graph
    Integrates different group-level representations
    """
    def __init__(self, mix_type='mean', n_mix=3, out_channels=5, pred_seq_len=12):
        super().__init__()
        self.mix_type = mix_type
        self.pred_seq_len = pred_seq_len
        if mix_type == 'mlp':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * pred_seq_len * n_mix, out_channels * pred_seq_len,
                                                       kernel_size=1), )
        elif mix_type == 'cnn':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * n_mix, out_channels,
                                                       kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, v_stack):
        n_batch, n_ped = v_stack[0].shape[0], v_stack[0].shape[3]
        if self.mix_type == 'sum':
            v = torch.stack(v_stack, dim=0).sum(dim=0)
        elif self.mix_type == 'mean':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
        elif self.mix_type == 'mlp':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v_stack = torch.cat(v_stack, dim=1).reshape(n_batch, -1, 1, n_ped)
            v = v + self.st_gcns_mix(v_stack).view(n_batch, -1, self.pred_seq_len, n_ped)
        elif self.mix_type == 'cnn':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v = v + self.st_gcns_mix(torch.cat(v_stack, dim=1))
        else:
            raise NotImplementedError
        return v


class DensityGroupFeatureExtractor(nn.Module):
    """
    Feature Extractor for Local Density and Group Size
    Extracts additional features as mentioned in the meeting
    """
    def __init__(self, density_radius=2.0, group_size_threshold=2):
        super().__init__()
        self.density_radius = density_radius
        self.group_size_threshold = group_size_threshold
        
    def compute_local_density(self, positions, radius=None):
        """
        Compute local density around each pedestrian
        Args:
            positions: (batch, seq_len, num_ped, 2) - absolute positions
            radius: density computation radius
        Returns:
            density: (batch, seq_len, num_ped) - local density values
        """
        if radius is None:
            radius = self.density_radius
            
        batch, seq_len, num_ped, _ = positions.shape
        density = torch.zeros(batch, seq_len, num_ped, device=positions.device)
        
        for b in range(batch):
            for t in range(seq_len):
                pos = positions[b, t]  # (num_ped, 2)
                for i in range(num_ped):
                    # Compute distance to all other pedestrians
                    distances = torch.norm(pos - pos[i].unsqueeze(0), dim=1)
                    # Count pedestrians within radius
                    density[b, t, i] = (distances < radius).sum().float() - 1  # -1 to exclude self
                    
        return density
    
    def compute_group_size(self, group_indices):
        """
        Compute group size for each pedestrian
        Args:
            group_indices: (num_ped,) - group assignment for each pedestrian
        Returns:
            group_sizes: (num_ped,) - size of group each pedestrian belongs to
        """
        unique_groups, counts = torch.unique(group_indices, return_counts=True)
        group_sizes = torch.zeros_like(group_indices, dtype=torch.float)
        
        for group_id, size in zip(unique_groups, counts):
            mask = group_indices == group_id
            group_sizes[mask] = size.float()
            
        return group_sizes
    
    def forward(self, positions, group_indices=None):
        """
        Extract density and group size features
        Args:
            positions: (batch, seq_len, num_ped, 2) - absolute positions
            group_indices: (num_ped,) - group assignment (optional)
        Returns:
            features: dict containing 'density' and 'group_size'
        """
        features = {}
        
        # Compute local density
        features['density'] = self.compute_local_density(positions)
        
        # Compute group size if group_indices provided
        if group_indices is not None:
            features['group_size'] = self.compute_group_size(group_indices)
        else:
            features['group_size'] = torch.ones(positions.shape[2], device=positions.device)
            
        return features


def generate_adjacency_matrix(v, mask=None):
    """Generate adjacency matrix for Social-STGCNN baseline"""
    n_ped = v.size(-1)
    temp = v.permute(0, 2, 3, 1).unsqueeze(dim=-2).repeat_interleave(repeats=n_ped, dim=-2)
    a = (temp - temp.transpose(2, 3)).norm(p=2, dim=-1)
    # inverse kernel
    a_inv = 1. / a
    a_inv[a == 0] = 0
    # masking
    a_inv = a_inv if mask is None else a_inv * mask
    # normalize
    a_hat = a_inv + torch.eye(n=n_ped, device=v.device)
    node_degrees = a_hat.sum(dim=-1).unsqueeze(dim=-1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0
    norm_degs_matrix = torch.eye(n=n_ped, device=v.device) * degs_inv_sqrt
    return torch.eye(n=n_ped, device=v.device) - norm_degs_matrix @ a_hat @ norm_degs_matrix


def generate_identity_matrix(v):
    """Generate spatial and temporal identity matrix for SGCN baseline"""
    i = [torch.eye(v.size(3), device=v.device).repeat(v.size(2), 1, 1),
         torch.eye(v.size(2), device=v.device).repeat(v.size(3), 1, 1)]
    return i
