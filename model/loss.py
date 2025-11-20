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


def multivariate_loss(V_pred, V_trgt, training=False):
    r"""Batch multivariate loss"""

    mu = V_trgt[:, :, :, 0:2] - V_pred[:, :, :, 0:2]
    mu = mu.unsqueeze(dim=-1)

    sx = V_pred[:, :, :, 2].exp()
    sy = V_pred[:, :, :, 3].exp()
    corr = V_pred[:, :, :, 4].tanh()

    cov = torch.zeros(V_pred.size(0), V_pred.size(1), V_pred.size(2), 2, 2).cuda()

    cov[:, :, :, 0, 0] = sx * sx
    cov[:, :, :, 0, 1] = corr * sx * sy
    cov[:, :, :, 1, 0] = corr * sx * sy
    cov[:, :, :, 1, 1] = sy * sy
    #cov = cov.clamp(min=-1e5, max=1e5)

    ### 스탠포드 데이터 넣으면서 reg, cov 조절
    # Add regularization to prevent singular matrix
    reg = torch.eye(2).cuda() * 1e-3
    # reg를 cov와 같은 차원으로 확장: [batch, seq_len, num_peds, 2, 2]
    reg = reg.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(cov)
    cov = cov + reg

    # 더 안전한 역행렬 계산: torch.linalg.pinv 사용 (singular matrix에도 안전)
    try:
        cov_inv = torch.linalg.pinv(cov)
    except:
        # fallback: 기존 방식
        cov_inv = cov.inverse()
    
    pdf = torch.exp(-0.5 * mu.transpose(-2, -1) @ cov_inv @ mu)

    # 디버그: 차원 확인
    #print(f"DEBUG LOSS: pdf.shape before squeeze = {pdf.shape}")
    det_cov = cov.det().clamp(min=1e-12)
    #print(f"DEBUG LOSS: cov.det().shape = {det_cov.shape}")

    # 마지막 두 개의 singleton 차원만 제거 (안전한 squeeze)
    pdf = pdf.squeeze(-1).squeeze(-1) / torch.sqrt(((2 * np.pi) ** 2) * det_cov)

    # NaN/Inf 처리: training과 validation 모두에서 처리
    pdf[torch.isinf(pdf) | torch.isnan(pdf)] = 0

    epsilon = 1e-20
    loss = -pdf.clamp(min=epsilon).log()
    
    # Loss 자체에도 NaN이 있는지 확인하고 처리
    loss[torch.isnan(loss) | torch.isinf(loss)] = 0

    return loss.mean()
