# loss.py
import torch

def multi_type_fcdd_loss(Y, T, eps=1e-6):
    B, C, H, W = Y.shape
    normal_term = Y.mean(dim=[2, 3], keepdim=True)
    loss = 0.0
    for i in range(C):
        anomaly_term = torch.log(1 - torch.exp(-normal_term[:, i:i+1, :, :]) + eps)
        is_good = 1 - T[:, i:i+1, :, :]
        term = is_good * normal_term[:, i:i+1, :, :] - (1 - is_good) * anomaly_term
        loss += term.mean()
    return loss