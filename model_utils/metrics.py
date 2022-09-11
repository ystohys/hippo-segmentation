import sys
import numpy as np
from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt
import torch
from torch import nn


def get_dice(tp, fp, fn):
    dice = (2 * tp) / ((2 * tp) + fp + fn)
    return dice


def get_precision(tp, fp):
    precision = tp / (tp + fp)
    return precision


def get_recall(tp, fn):
    recall = tp / (tp + fn)
    return recall

def get_train_time(hist, per_epoch=True, kfcv=False):
    if kfcv:
        per_fold = []
        if per_epoch:
            for fold in hist['time_elapsed_epoch']:
                per_fold.append(fold[-1]/len(fold))
            return np.mean(per_fold)
        else:
            for fold in hist['time_elapsed_epoch']:
                per_fold.append(fold[-1])
            return np.mean(per_fold)
    else:
        if per_epoch:
            avg_time = hist['time_elapsed_epoch'][-1] / len(hist['time_elapsed_epoch'])
            return avg_time
        else:
            return hist['time_elapsed_epoch'][-1]


def per_slice_dice_stats(
        pred,
        tgt
):
    pred_slice = pred.detach().clone()
    tgt_slice = tgt.detach().clone()
    batch_size = pred_slice.size(0)

    pred_slice = torch.sigmoid(pred_slice)
    pred_slice = (pred_slice >= 0.5).type(torch.float32)

    pred_flat = pred_slice.contiguous().view(batch_size, -1)
    tgt_flat = tgt_slice.contiguous().view(batch_size, -1)

    true_positives = (pred_flat * tgt_flat).sum(dim=1)
    false_positives = pred_flat.sum(dim=1) - true_positives
    false_negatives = tgt_flat.sum(dim=1) - true_positives

    return true_positives, false_positives, false_negatives


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


def batch_precision(
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

    eps = sys.float_info.epsilon  # To prevent division by zero

    total_precision_tensor = (intersection) / (pred_flat_sum + eps)
    mean_precision_tensor = total_precision_tensor.mean()
    mean_precision = mean_precision_tensor.item()

    if mean_precision < 0.0 or mean_precision > 1.0:
        raise ValueError("Precision is not between 0 and 1!")

    return mean_precision


def batch_recall(
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
    tgt_flat_sum = tgt_flat.sum(dim=1)  # True positives + False negatives

    eps = sys.float_info.epsilon  # To prevent division by zero

    total_recall_tensor = (intersection) / (tgt_flat_sum + eps)
    mean_recall_tensor = total_recall_tensor.mean()
    mean_recall = mean_recall_tensor.item()

    if mean_recall < 0.0 or mean_recall > 1.0:
        raise ValueError("Recall is not between 0 and 1!")

    return mean_recall


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


def plot_val_per_epoch(hist_dict, metrics, ylabel, title, ylims):
    if isinstance(metrics, tuple):
        metric_mean = np.mean(hist_dict[metrics[0]], axis=0)
        metric_std = np.std(hist_dict[metrics[0]], axis=0)
        fig, ax = plt.subplots()
        p1 = ax.plot(np.arange(len(metric_mean)), metric_mean)
        p2 = ax.fill(np.NaN, np.NaN, 'orange', alpha=0.5)
        ax.fill_between(x=np.arange(len(metric_mean)),
                        y1=metric_mean+metric_std,
                        y2=metric_mean-metric_std,
                        alpha=0.5)
        ax.set_ylim(ylims)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Epoch')
        ax.legend([metrics[1]], loc='upper left')
        plt.show()
    elif isinstance(metrics, list):
        legend_obj = []
        legends = []
        fig, ax = plt.subplots()
        best_val = 0
        for m in metrics:
            metric_mean = np.mean(hist_dict[m[0]], axis=0)
            metric_std = np.std(hist_dict[m[0]], axis=0)
            p1 = ax.plot(np.arange(len(metric_mean)), metric_mean)
            p2 = ax.fill(np.NaN, np.NaN, p1[0].get_color(), alpha=0.5)
            ax.fill_between(x=np.arange(len(metric_mean)),
                            y1=metric_mean + metric_std,
                            y2=metric_mean - metric_std,
                            alpha=0.5)
            if 'loss' in m[0] and m[1] == 'val':
                best_val = min(metric_mean)
                ax.axhline(y=best_val, 
                           xmin=0, 
                           #xmax=np.argmin(metric_mean)/len(metric_mean), 
                           xmax = len(metric_mean),
                           color='red', 
                           linestyle='dashed')
                trans = blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
                ax.text(x=0, y=best_val, s='{0:.4f}'.format(best_val), ha='right', va='center', c='red', transform=trans)
            elif 'dice' in m[0] and m[1] == 'val':
                best_val = max(metric_mean)
                ax.axhline(y=best_val, 
                           xmin=0, 
                           #xmax=np.argmax(metric_mean)/len(metric_mean), 
                           xmax=len(metric_mean),
                           color='red', 
                           linestyle='dashed')
                trans = blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
                ax.text(x=0, y=best_val, s='{0:.4f}'.format(best_val), ha='right', va='center', c='red', transform=trans)
            legend_obj.append((p2[0], p1[0]))
            legends.append(m[1])
        ax.set_ylim(ylims)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Epoch')
        ax.legend(legend_obj, legends, loc='upper left')
        plt.show()



