import os
import collections
import datetime
import functools
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import nibabel as nib
from tqdm.auto import tqdm
import torch.cuda
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_utils.dataset import HarpDataset
from model_utils.metrics import batch_dice_metric, per_slice_dice_stats


VIEW_SLICES = {
    0: 40,
    1: 56,
    2: 72
}


def start_eval(
        model,
        dir_name,
        brain_side,
        test_ids,
        batch_size,
        verbose=False
):
    """
    Evaluates the model on a train/validation set, and returns the total loss (based on the entire dataset) and the
    average metric
    :param model:
    :param dir_name:
    :param brain_side:
    :param test_ids:
    :param batch_size:
    :param verbose:
    :return:
    """
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

    mean_loss = np.mean(total_loss)  # Average loss per subject
    mean_metric = np.mean(total_metric)  # Average metric per subject
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
        running_loss = []
        epoch_metric = collections.deque([])
        for i, data in enumerate(train_loader):
            mri_vol, hip_label = data
            mri_vol, hip_label = mri_vol.to(device), hip_label.to(device)
            optimizer.zero_grad()

            hip_pred = model(mri_vol)
            loss = loss_func(hip_pred, hip_label)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            epoch_metric.append(batch_dice_metric(hip_pred, hip_label))

        epoch_end = datetime.datetime.now()
        history['train_loss_per_epoch'][epoch] = np.mean(running_loss)  # Average (per subject) train loss per epoch
        history['train_metric_per_epoch'][epoch] = np.mean(epoch_metric)  # Average (per subject) DICE score per epoch
        history['time_elapsed_epoch'][epoch] = (epoch_end - start_time).total_seconds()
        pbar.set_description(
            'Epoch: {0}, Train Loss: {1:.5f}, Train Metric: {2:.5f}'.format(epoch+1,
                                                                            history['train_loss_per_epoch'][epoch],
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
        running_loss = []
        epoch_metric = collections.deque([])
        for i, data in enumerate(train_loader):
            mri_vol, hip_label = data
            mri_vol, hip_label = mri_vol.to(device), hip_label.to(device)
            optimizer.zero_grad()

            hip_pred = model(mri_vol)
            loss = loss_func(hip_pred, hip_label)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            epoch_metric.append(batch_dice_metric(hip_pred, hip_label))

        epoch_end = datetime.datetime.now()
        val_loss, val_metric = start_eval(
            model,
            dir_name,
            brain_side,
            val_ids,
            batch_size
        )
        history['train_loss_per_epoch'][epoch] = np.mean(running_loss)
        history['train_metric_per_epoch'][epoch] = np.mean(epoch_metric)
        history['val_loss_per_epoch'][epoch] = val_loss
        history['val_metric_per_epoch'][epoch] = val_metric
        history['time_elapsed_epoch'][epoch] = (epoch_end - start_time).total_seconds()
        pbar.set_description('Epoch: {0}, '.format(epoch+1))
        pbar.set_postfix_str(
            ('Train Loss: {0:.5f}, Train Metric: {1:.5f}, '
             'Validation Loss: {2:.5f}, Validation Metric: {3:.5f}').format(history['train_loss_per_epoch'][epoch],
                                                                            history['train_metric_per_epoch'][epoch],
                                                                            history['val_loss_per_epoch'][epoch],
                                                                            history['val_metric_per_epoch'][epoch])
        )

    return history


def skfcv_train_model(
        model_class,  # Must be model class, not actual model instance
        dir_name,
        brain_side,
        meta_file,
        train_ids,
        transforms,
        batch_size,
        num_epochs,
        learning_rate,
        kfold=5,
        random_seed=42
):
    harp_meta = pd.read_csv(meta_file)
    train_meta = harp_meta.loc[harp_meta['Subject'].isin(train_ids), ['Subject', 'Group', 'Age', 'Sex']]
    skf = StratifiedKFold(n_splits=kfold, random_state=random_seed, shuffle=True)
    total_hist = {
        'train_loss_per_epoch': [],
        'train_metric_per_epoch': [],
        'val_loss_per_epoch': [],
        'val_metric_per_epoch': [],
        'time_elapsed_epoch': []
    }
    for fold_num, train_test_idx in enumerate(skf.split(train_meta['Subject'], train_meta['Group'])):
        print('Fold {0}'.format(fold_num+1))
        tr_train_ids = list(train_meta.iloc[train_test_idx[0], :].loc[:, 'Subject'])
        tr_val_ids = list(train_meta.iloc[train_test_idx[1], :].loc[:, 'Subject'])
        tmp_model = model_class(1)
        fold_history = hocv_train_model(tmp_model,
                                        dir_name,
                                        brain_side,
                                        tr_train_ids,
                                        tr_val_ids,
                                        transforms,
                                        batch_size,
                                        num_epochs,
                                        learning_rate)
        for i in total_hist.keys():
            total_hist[i].append(fold_history[i])

    for s in total_hist.keys():
        total_hist[s] = np.array(total_hist[s])

    return total_hist


def save_model(model, save_path=None):
    if not save_path:
        save_path = 'model_{0}.pth'.format(datetime.datetime.now().strftime('%d%m%Y_%H%M%S'))
    torch.save(model.state_dict(), save_path)
    print('Model saved at: {0}'.format(save_path))


def model_predict(
        model,
        dir_name,
        subj_id,
        brain_side,
        model_file=None,
        dv='cpu'
):
    if isinstance(model_file, str):
        model.load_state_dict(torch.load(model_file, map_location=torch.device(dv)))
    model.eval()

    subj = nib.load(os.path.join(dir_name, subj_id, "SMALL_{0}_{1}B.nii".format(subj_id, brain_side.upper())))
    subj_arr = subj.get_fdata()
    subj_tensor = torch.tensor(subj_arr, dtype=torch.float32)
    subj_tensor = torch.unsqueeze(torch.unsqueeze(subj_tensor, 0), 0)

    with torch.no_grad():
        predicted = model(subj_tensor)
        predicted = torch.sigmoid(predicted)
        predicted = (predicted >= 0.5).type(torch.float32)
        predicted = torch.squeeze(predicted)
        pred_arr = predicted.detach().numpy()

    model.train()
    pred_nifti = nib.Nifti1Image(pred_arr, subj.affine)
    nib.save(pred_nifti, os.path.join(dir_name, subj_id, "PRED_{0}H_{1}.nii".format(brain_side.upper(), subj_id)))


############################################
#  Utility functions for 2D USegNet models #
############################################


def start_2d_eval(
        model,
        view,
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

    mean_loss = np.mean(total_loss)  # Average loss per SLICE
    mean_metric = np.mean(total_metric)  # Average metric per SLICE
    model.train()  # Turns model back into training mode
    if verbose:
        print('Average loss: {0:.5f}, Average metric: {1:.5f}'.format(mean_loss, mean_metric))
    return mean_loss, mean_metric


def train_2d_model(
        model,
        view,
        dir_name,
        brain_side,
        train_ids,
        batch_size,
        num_epochs,
        learning_rate=0.001
):
    """
    Trains the model passed on training set (no cross validation done). Model is trained in_place
    :param model: Model object (2D)
    :param view: Takes a value of 0, 1 or 2 which corresponds to the orthogonal plane we are training on
    :param dir_name: Name of directory where all MRI volumes and masks are stored
    :param brain_side: Side of brain that we are analysing
    :param train_ids: Subject IDs of MRI volumes and masks belonging to training set
    :param batch_size: Batch size for training
    :param num_epochs: Number of training epochs
    :param learning_rate: Learning rate for optimizer function
    :return: History of training metric, loss, time taken
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    harp_dataset = HarpDataset(dir_name, brain_side)
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
        running_loss = []
        epoch_metric = collections.deque([])
        for i, data in enumerate(train_loader):
            per_subject_loss = 0  # For one subject
            tp, fp, fn = torch.zeros(batch_size), torch.zeros(batch_size), torch.zeros(batch_size)
            tp, fp, fn = tp.to(device), fp.to(device), fn.to(device)
            mri_vol, hip_label = data
            mri_vol, hip_label = mri_vol.to(device), hip_label.to(device)
            for slice_idx in range(VIEW_SLICES[view]):
                optimizer.zero_grad()
                if view == 0:
                    mri_vol_slice = mri_vol[:,:,slice_idx,:,:]
                    hip_lab_slice = hip_label[:,:,slice_idx,:,:]
                elif view == 1:
                    mri_vol_slice = mri_vol[:,:,:,slice_idx,:]
                    hip_lab_slice = hip_label[:,:,:,slice_idx,:]
                elif view == 2:
                    mri_vol_slice = mri_vol[:,:,:,:,slice_idx]
                    hip_lab_slice = hip_label[:,:,:,:,slice_idx]
                hip_pred = model(mri_vol_slice)
                loss = loss_func(hip_pred, hip_lab_slice)
                loss.backward()
                optimizer.step()
                per_subject_loss += loss.item()
                slice_tp, slice_fp, slice_fn = per_slice_dice_stats(hip_pred, hip_lab_slice)
                slice_tp, slice_fp, slice_fn = slice_tp.to(device), slice_fp.to(device), slice_fn.to(device)
                tp += slice_tp
                fp += slice_fp
                fn += slice_fn
            running_loss.append(per_subject_loss)
            total_dice = (2 * tp) / ((2*tp) + fp + fn)
            per_subject_dice = total_dice.mean()
            epoch_metric.append(per_subject_dice.item())

        epoch_end = datetime.datetime.now()
        history['train_loss_per_epoch'][epoch] = np.mean(running_loss)  # Average (per subject) train loss per epoch
        history['train_metric_per_epoch'][epoch] = np.mean(epoch_metric)
        history['time_elapsed_epoch'][epoch] = (epoch_end - start_time).total_seconds()
        pbar.set_description(
            'Epoch: {0}, Train Loss: {1:.5f}, Train Metric: {2:.5f}'.format(epoch+1,
                                                                            history['train_loss_per_epoch'][epoch],
                                                                            history['train_metric_per_epoch'][epoch]
                                                                            )
        )

    return history


def hocv_train_2d_model(
        model,
        view,
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
        running_loss = []
        epoch_metric = collections.deque([])
        for i, data in enumerate(train_loader):
            mri_vol, hip_label = data
            mri_vol, hip_label = mri_vol.to(device), hip_label.to(device)
            for slice_idx in range(VIEW_SLICES[view]):
                optimizer.zero_grad()
                if view == 0:
                    mri_vol_slice = mri_vol[:, :, slice_idx, :, :]
                    hip_lab_slice = hip_label[:, :, slice_idx, :, :]
                elif view == 1:
                    mri_vol_slice = mri_vol[:, :, :, slice_idx, :]
                    hip_lab_slice = hip_label[:, :, :, slice_idx, :]
                elif view == 2:
                    mri_vol_slice = mri_vol[:, :, :, :, slice_idx]
                    hip_lab_slice = hip_label[:, :, :, :, slice_idx]
                hip_pred = model(mri_vol_slice)
                loss = loss_func(hip_pred, hip_lab_slice)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                epoch_metric.append(batch_dice_metric(hip_pred, hip_lab_slice))

        epoch_end = datetime.datetime.now()
        val_loss, val_metric = start_2d_eval(
            model,
            dir_name,
            brain_side,
            val_ids,
            batch_size
        )
        history['train_loss_per_epoch'][epoch] = np.mean(running_loss)  # Average loss per SLICE
        history['train_metric_per_epoch'][epoch] = np.mean(epoch_metric)  # Average metric per SLICE
        history['val_loss_per_epoch'][epoch] = val_loss
        history['val_metric_per_epoch'][epoch] = val_metric
        history['time_elapsed_epoch'][epoch] = (epoch_end - start_time).total_seconds()
        pbar.set_description('Epoch: {0}, '.format(epoch+1))
        pbar.set_postfix_str(
            ('Train Loss: {0:.5f}, Train Metric: {1:.5f}, '
             'Validation Loss: {2:.5f}, Validation Metric: {3:.5f}').format(history['train_loss_per_epoch'][epoch],
                                                                            history['train_metric_per_epoch'][epoch],
                                                                            history['val_loss_per_epoch'][epoch],
                                                                            history['val_metric_per_epoch'][epoch])
        )

    return history

