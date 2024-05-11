import torch
from torch import nn


def masked_loss(output, target):
    mask = torch.isfinite(target)
    masked_output = output[mask]
    masked_target = target[mask]
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(masked_output, masked_target)
    return loss


def masked_loss_nan(output, target):
    mask = torch.isfinite(target)
    masked_target = target.clone()
    masked_target[~mask] = 0

    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(output, masked_target)

    loss = loss * mask.float()
    return loss


def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target))


def mspe_loss(output, target):
    return torch.mean(torch.square((target - output) / target))
