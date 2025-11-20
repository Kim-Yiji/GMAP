import numpy as np
import torch


def generate_statistics_matrices(V):
    r"""generate mean and covariance matrices from the network output."""
    
    # 4차원 텐서 처리 - (batch_size, num_peds, seq_len, 5)
    mu = V[:, :, :, 0:2]
    sx = V[:, :, :, 2].exp()
    sy = V[:, :, :, 3].exp()
    corr = V[:, :, :, 4].tanh()
    cov = torch.zeros(V.size(0), V.size(1), V.size(2), 2, 2).cuda()

    cov[:, :, :, 0, 0] = sx * sx
    cov[:, :, :, 0, 1] = corr * sx * sy
    cov[:, :, :, 1, 0] = corr * sx * sy
    cov[:, :, :, 1, 1] = sy * sy

    return mu, cov


def _tensor_stats(tensor):
    with torch.no_grad():
        flat = tensor.detach().view(-1)
        if flat.numel() == 0:
            return "empty"
        return f"min={flat.min().item():.3e}, max={flat.max().item():.3e}, mean={flat.mean().item():.3e}"


def _log_loss_diagnostics(reason, mu, sx, sy, corr, nll, invalid_mask):
    try:
        msg = [
            f"[LossDiag] reason={reason}",
            f"nll_max={nll.max().item():.3e}",
            f"nll_min={nll.min().item():.3e}",
            f"invalid_count={invalid_mask.sum().item()}",
            f"mu:{_tensor_stats(mu)}",
            f"sx:{_tensor_stats(sx)}",
            f"sy:{_tensor_stats(sy)}",
            f"corr:{_tensor_stats(corr)}",
        ]
        print(" ".join(msg), flush=True)
    except Exception:
        # Logging must never break training
        pass


def multivariate_loss(V_pred, V_trgt, training=False):
    r"""Batch multivariate loss in log-domain for numerical stability"""

    device = V_pred.device
    dtype = V_pred.dtype

    mu = V_trgt[:, :, :, 0:2] - V_pred[:, :, :, 0:2]
    # Clamp mu to prevent gradient explosion
    mu = mu.clamp(min=-100.0, max=100.0)
    mu = mu.unsqueeze(dim=-1)  # (..., 2, 1)

    min_sigma = 1e-2
    max_sigma = 10.0
    sx = torch.nn.functional.softplus(V_pred[:, :, :, 2]) + min_sigma
    sy = torch.nn.functional.softplus(V_pred[:, :, :, 3]) + min_sigma
    sx = sx.clamp(max=max_sigma)
    sy = sy.clamp(max=max_sigma)
    corr = V_pred[:, :, :, 4].tanh().clamp(min=-0.99, max=0.99)

    cov = torch.zeros(V_pred.size(0), V_pred.size(1), V_pred.size(2), 2, 2, device=device, dtype=dtype)

    cov[:, :, :, 0, 0] = sx * sx
    cov[:, :, :, 1, 1] = sy * sy
    off_diag = corr * sx * sy
    cov[:, :, :, 0, 1] = off_diag
    cov[:, :, :, 1, 0] = off_diag

    eye = torch.eye(2, device=device, dtype=dtype)
    jitter = 1e-3
    max_tries = 3

    chol, info = torch.linalg.cholesky_ex(cov)
    attempt = 0
    while info.any() and attempt < max_tries:
        bad = info > 0
        cov = torch.where(bad.unsqueeze(-1).unsqueeze(-1), cov + eye * jitter, cov)
        chol, info = torch.linalg.cholesky_ex(cov)
        jitter *= 10
        attempt += 1

    if info.any():
        # Fallback to diagonal covariance (ignore correlation) for the problematic entries
        diag_cov = torch.zeros_like(cov)
        diag_cov[:, :, :, 0, 0] = (sx * sx) + min_sigma
        diag_cov[:, :, :, 1, 1] = (sy * sy) + min_sigma
        cov = torch.where(info.unsqueeze(-1).unsqueeze(-1) > 0, diag_cov + eye * jitter, cov)
        chol = torch.linalg.cholesky(cov)

    # Solve for Mahalanobis distance using the Cholesky factors
    solved = torch.cholesky_solve(mu, chol)
    mahalanobis = (mu.transpose(-2, -1) @ solved).squeeze(-1).squeeze(-1)

    log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
    log_two_pi = np.log(2 * np.pi)
    log_norm_const = 2.0 * log_two_pi

    nll = 0.5 * (mahalanobis + log_det + log_norm_const)
    invalid_mask = torch.isnan(nll) | torch.isinf(nll)

    if training:
        if invalid_mask.any():
            _log_loss_diagnostics("invalid", mu, sx, sy, corr, nll, invalid_mask)
        else:
            large_mask = nll > 50.0
            if large_mask.any():
                _log_loss_diagnostics("large_nll", mu, sx, sy, corr, nll, large_mask)

    if invalid_mask.all():
        raise ValueError("All entries in negative log-likelihood became invalid.")
    if invalid_mask.any():
        nll = nll[~invalid_mask]

    return nll.mean()
