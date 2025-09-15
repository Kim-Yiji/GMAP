# Utility functions for model components
# Based on DMRGCN utilities with device-agnostic improvements

import torch
import torch.nn as nn


def normalized_adjacency_matrix(A):
    """Returns the normalized Adjacency matrix."""
    device = A.device
    node_degrees = A.sum(-1).unsqueeze(dim=-1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0
    norm_degs_matrix = torch.eye(A.size(-1), device=device) * degs_inv_sqrt
    return norm_degs_matrix @ A @ norm_degs_matrix


def normalized_adjacency_tilde_matrix(A):
    """Returns the normalized Adjacency tilde (A~) matrix."""
    device = A.device
    A_t = A + torch.eye(A.size(-1), device=device)
    return normalized_adjacency_matrix(A_t)


def normalized_laplacian_matrix(A):
    """Returns the normalized Laplacian matrix."""
    device = A.device
    return torch.eye(A.size(-1), device=device) - normalized_adjacency_matrix(A)


def normalized_laplacian_tilde_matrix(A):
    """Returns the normalized Laplacian tilde (L~) matrix."""
    device = A.device
    A_t = A + torch.eye(A.size(-1), device=device)
    return torch.eye(A_t.size(-1), device=device) - normalized_adjacency_matrix(A_t)


def drop_edge(A, percent, training=True, inplace=False):
    """Returns the randomly dropped edge Adjacency matrix with preserve rate."""
    assert 0 <= percent <= 1.0
    if not training:
        return A
    A_prime = torch.rand_like(A)
    A_drop = A if inplace else A.clone()
    A_drop[A_prime > percent] = 0
    return A_drop


def clip_adjacency_matrix(A, min_val=-1e10, max_val=1e10):
    """Returns the clipped Adjacency matrix with min and max values."""
    A_c = A.clamp(min=min_val, max=max_val)
    A_c[A_c == min_val] = 0
    A_c[A_c == max_val] = 0
    A_c[A_c > 0] = 1
    return A_c


def get_disentangled_adjacency_matrix(A, split=[]):
    """Returns the list of clipped Adjacency matrix split by list values."""
    if len(split) == 0:
        return A.unsqueeze(1)  # Add relation dimension
    
    split_sorted = sorted(split)
    split_sorted = split_sorted + [1e10]
    
    A_d = []
    for i in range(len(split_sorted) - 1):
        A_d.append(clip_adjacency_matrix(A, min_val=split_sorted[i], max_val=split_sorted[i + 1]))
    
    return torch.stack(A_d, dim=1)  # (B, R, T, N, N)
