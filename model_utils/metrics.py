import sys
import numpy as np
import matplotlib.pyplot as plt
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


def plot_per_epoch(hist_dict, metrics, ylabel, title):
    if not isinstance(metrics, (tuple, list)):
        raise TypeError(
            "metrics argument must be passed as a tuple (for one metric) or a list of tuples (for multiple metrics)"
        )
    if isinstance(metrics, tuple):
        plt.plot(hist_dict[metrics[0]])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend([metrics[1]], loc='upper left')
        plt.show()
    elif isinstance(metrics, list):
        legends = []
        for m in metrics:
            plt.plot(hist_dict[m[0]])
            legends.append(m[1])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend(legends, loc='upper left')
        plt.show()


def plot_val_per_epoch(hist_dict, metrics, ylabel, title):
    if isinstance(metrics, tuple):
        metric_mean = np.mean(hist_dict[metrics[0]], axis=0)
        metric_std = np.std(hist_dict[metrics[0]], axis=0)
        plt.plot(np.arange(len(metric_mean), metric_mean))
        plt.fill_between(np.arange(len(metric_mean)), metric_mean+metric_std, metric_mean-metric_std)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.legend([metrics[1]], loc='upper left')
        plt.show()



