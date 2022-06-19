import collections
import datetime
import functools
import logging
import numpy as np
import tqdm
import torch.cuda
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_utils.dataset import HarpDataset
from model_utils.metrics import batch_dice_metric


def train_model(
        model,
        dir_name,
        brain_side,
        train_ids,
        transforms,
        num_epochs,
        learning_rate=0.001
):
    """
    Trains the model passed on training set (no cross validation done). Model is trained in_place
    :param model: Model object
    :param dir_name: Name of directory where all MRI volumes and masks are stored
    :param brain_side: Side of brain that we are analysing
    :param train_ids: Subject IDs of MRI volumes and masks belonging to training set
    :param transforms: Transforms to apply for data augmentation to training images
    :param num_epochs: Number of training epochs
    :param learning_rate: Learning rate for optimizer function
    :return: History of training metric, loss, time taken
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    harp_dataset = HarpDataset(dir_name, brain_side, transforms)
    id_sampler = SubsetRandomSampler(train_ids)
    train_loader = DataLoader(
        dataset=harp_dataset,
        sampler=id_sampler,
        batch_size=2
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.BCEWithLogitsLoss()
    history = {
        'loss_per_epoch': np.zeros(num_epochs),
        'metric_per_epoch': np.zeros(num_epochs),
        'time_elapsed_epoch': np.zeros(num_epochs)
    }
    start_time = datetime.datetime.now()
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        running_loss = 0.0
        epoch_metric = collections.deque([])
        for i, data in enumerate(train_loader):
            mri_vol, hip_label = data
            mri_vol, hip_label = mri_vol.to(device), hip_label.to(device)
            optimizer.zero_grad()

            hip_pred = model(mri_vol)
            loss = loss_func(hip_pred, hip_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_metric.append(batch_dice_metric(hip_pred, hip_label))

        epoch_end = datetime.datetime.now()
        history['loss_per_epoch'][epoch] = running_loss
        history['metric_per_epoch'][epoch] = np.mean(epoch_metric)
        history['time_elapsed_epoch'][epoch] = (epoch_end - start_time).total_seconds()
        pbar.set_description(
            'Epoch: {0}, Train Loss: {1:.5f}, Train Metric: {2:.5f}'.format(epoch+1,
                                                                            running_loss,
                                                                            history['metric_per_epoch'][epoch]
                                                                            )
        )

    return history


def hocv_train_model(
        model,
        dir_name,
        train_ids,
        transforms,
        num_epochs,
        learning_rate
):
    pass


def skfcv_train_model(
        model,
        dir_name,
        train_ids,
        transforms,
        num_epochs,
        learning_rate
):
    pass

