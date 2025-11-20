import torch
import math


def normalize_features(z, eps=1e-6):
    """Apply simple per-component normalization/clipping for affordance features.

    z: (..., 4) with components [d_ij, v_plus_ij, size_j, ttc_ij]
    """
    d = torch.clamp(z[..., 0], 0.0, 10.0) / 4.0
    v = torch.clamp(z[..., 1], 0.0, 10.0) / 2.0
    s = torch.clamp(z[..., 2], 0.0, 1e4) / 500.0
    ttc = torch.clamp(z[..., 3], 0.0, 10.0) / 4.0
    return torch.stack([d, v, s, ttc], dim=-1)


def compute_threat_scores(pos_t, vel_t, sizes, ped_mask, obj_mask, eps=1e-3, weights=None):
    """Compute threat score matrices per time step.

    Args:
        pos_t: (V, 2) absolute positions at time t
        vel_t: (V, 2) velocities at time t
        sizes: (V,) bbox areas (0 for pedestrians if unknown)
        ped_mask: (V,) bool mask for pedestrians
        obj_mask: (V,) bool mask for dynamic obstacles
        eps: small constant
        weights: optional (4,) tensor for weighted sum; if None, use [1,1,1,1]

    Returns:
        T_pp: (V, V) PP threat scores in [0,1]
        T_po: (V, V) PO threat scores in [0,1] (i=ped, j=obj)
    """
    V = pos_t.size(0)
    diff = pos_t.unsqueeze(1) - pos_t.unsqueeze(0)  # (V,V,2)
    d_ij = torch.linalg.norm(diff, dim=-1)  # (V,V)
    rel_vel = vel_t.unsqueeze(1) - vel_t.unsqueeze(0)  # (V,V,2)
    approach = -(rel_vel * diff).sum(dim=-1) / (d_ij + eps)  # projection (positive if approaching)
    v_plus = torch.clamp(approach, min=0.0)
    size_j = sizes.unsqueeze(0).repeat(V, 1)  # (V,V)
    ttc = d_ij / torch.clamp(v_plus, min=eps)

    z = torch.stack([d_ij, v_plus, size_j, ttc], dim=-1)
    z = normalize_features(z)

    if weights is None:
        weights = torch.tensor([1.0, 1.0, 0.5, 1.0], device=z.device, dtype=z.dtype)
    logits = (z * weights).sum(dim=-1)
    threat = torch.sigmoid(logits)

    ped_mask_row = ped_mask.unsqueeze(1).float()
    ped_mask_col = ped_mask.unsqueeze(0).float()
    obj_mask_col = obj_mask.unsqueeze(0).float()

    T_pp = threat * (ped_mask_row * ped_mask_col)
    T_po = threat * (ped_mask_row * obj_mask_col)
    return T_pp, T_po


