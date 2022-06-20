import collections
import datetime
import functools
import logging
import numpy as np
from tqdm import tqdm
import torch.cuda
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_utils.dataset import HarpDataset
from model_utils.metrics import batch_dice_metric


def start_eval(
        model,
        dir_name,
        brain_side,
        test_ids,
        batch_size,
        verbose=False
):
    if model.training:
        model.eval()  # Set model to eval mode if not yet done so
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = HarpDataset(dir_name, brain_side)
    id_sampler = SubsetRandomSampler(test_ids)
    test_loader = DataLoader(test_set, sampler=id_sampler, batch_size=batch_size)
    total_loss = []
    total_metric = []
    loss_func = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for data in test_loader:
            mri_vol, hip_label = data
            mri_vol, hip_label = mri_vol.to(device), hip_label.to(device)
            hip_pred = model(mri_vol)
            total_loss.append(loss_func(hip_pred, hip_label).item())
            total_metric.append(batch_dice_metric(hip_pred, hip_label))

    mean_loss = np.mean(total_loss)
    mean_metric = np.mean(total_metric)
    model.train()  # Turns model back into training mode
    if verbose:
        print('Average loss: {0:.5f}, Average metric: {1:.5f}'.format(mean_loss, mean_metric))
    return mean_loss, mean_metric


def train_model(
        model,
        dir_name,
        brain_side,
        train_ids,
        transforms,
        batch_size,
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
    :param batch_size: Batch size for training
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
        batch_size=batch_size
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.BCEWithLogitsLoss()
    history = {
        'train_loss_per_epoch': np.zeros(num_epochs),
        'train_metric_per_epoch': np.zeros(num_epochs),
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
        history['train_loss_per_epoch'][epoch] = running_loss
        history['train_metric_per_epoch'][epoch] = np.mean(epoch_metric)
        history['time_elapsed_epoch'][epoch] = (epoch_end - start_time).total_seconds()
        pbar.set_description(
            'Epoch: {0}, Train Loss: {1:.5f}, Train Metric: {2:.5f}'.format(epoch+1,
                                                                            running_loss,
                                                                            history['train_metric_per_epoch'][epoch]
                                                                            )
        )

    return history


def hocv_train_model(
        model,
        dir_name,
        brain_side,
        train_ids,
        val_ids,
        transforms,
        batch_size,
        num_epochs,
        learning_rate
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    harp_dataset = HarpDataset(dir_name, brain_side, transforms)
    id_sampler = SubsetRandomSampler(train_ids)
    train_loader = DataLoader(
        dataset=harp_dataset,
        sampler=id_sampler,
        batch_size=batch_size
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.BCEWithLogitsLoss()
    history = {
        'train_loss_per_epoch': np.zeros(num_epochs),
        'train_metric_per_epoch': np.zeros(num_epochs),
        'val_loss_per_epoch': np.zeros(num_epochs),
        'val_metric_per_epoch': np.zeros(num_epochs),
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
        val_loss, val_metric = start_eval(model, dir_name, brain_side, batch_size, val_ids)
        history['train_loss_per_epoch'][epoch] = running_loss
        history['train_metric_per_epoch'][epoch] = np.mean(epoch_metric)
        history['val_loss_per_epoch'][epoch] = val_loss
        history['val_metric_per_epoch'][epoch] = val_metric
        history['time_elapsed_epoch'][epoch] = (epoch_end - start_time).total_seconds()
        pbar.set_description(
            ('Epoch: {0} \n'
             'Train Loss: {1:.5f}, Train Metric: {2:.5f}\n'
             'Validation Loss: {3:.5f}, Validation Metric: {4:.5f}').format(epoch+1,
                                                                            running_loss,
                                                                            history['train_metric_per_epoch'][epoch],
                                                                            history['val_loss_per_epoch'][epoch],
                                                                            history['val_metric_per_epoch'][epoch]
                                                                            )
        )

    return history


def skfcv_train_model(
        model,
        dir_name,
        train_ids,
        transforms,
        num_epochs,
        learning_rate
):
    pass

