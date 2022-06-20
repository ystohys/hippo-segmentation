import sys
import torch
from torch import nn


def batch_dice_metric(
        pred,  # Tensor of logits
        tgt
):
    pred_vol = pred.detach().clone()
    tgt_vol = tgt.detach().clone()
    batch_size = pred_vol.size(0)

    pred_vol = torch.sigmoid(pred_vol)
    pred_vol = (pred_vol >= 0.5).type(torch.float32)

    pred_flat = pred_vol.contiguous().view(batch_size, -1)
    tgt_flat = tgt_vol.contiguous().view(batch_size, -1)

    intersection = (pred_flat * tgt_flat).sum(dim=1)  # True positives
    pred_flat_sum = pred_flat.sum(dim=1)  # True positives + False positives
    tgt_flat_sum = tgt_flat.sum(dim=1)  # True positives + False negatives

    eps = sys.float_info.epsilon  # To prevent division by zero

    total_dice_tensor = (2.0 * intersection) / (pred_flat_sum + tgt_flat_sum + eps)
    mean_dice_tensor = total_dice_tensor.mean()
    mean_dice = mean_dice_tensor.item()

    if mean_dice < 0.0 or mean_dice > 1.0:
        raise ValueError("Dice score is not between 0 and 1!")

    return mean_dice




