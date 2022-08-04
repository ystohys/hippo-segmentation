import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchio as tio
from torchio.transforms import SpatialTransform, IntensityTransform


def get_train_test_subjects(
        meta_file,
        mri_path,
        random_seed,
        subj_only=True
):
    valid_subjects = os.listdir(mri_path)
    harp_meta = pd.read_csv(meta_file)
    harp_meta = harp_meta.loc[harp_meta['Subject'].isin(valid_subjects), ['Subject', 'Group', 'Age', 'Sex']]
    train_meta, test_meta = train_test_split(
        harp_meta,
        test_size=25,
        random_state=random_seed,
        shuffle=True,
        stratify=harp_meta['Group']
    )
    if subj_only:
        return list(train_meta['Subject']), list(test_meta['Subject'])
    else:
        return train_meta, test_meta


def split_train_subjects(
        meta_file,
        train_ids,
        num_of_val,
        random_seed
):
    """
    Further split the training data into training and validation sets for cross-validation.
    :param meta_file: CSV file containing metadata of subjects
    :param train_ids: Subject IDs of MRIs in training set
    :param num_of_val: Number of subjects to use for validation set
    :param random_seed: Put an integer for reproducible results
    :return: subject IDs of training and validation subjects
    """
    harp_meta = pd.read_csv(meta_file)
    harp_meta = harp_meta.loc[harp_meta['Subject'].isin(train_ids), ['Subject', 'Group', 'Age', 'Sex']]
    train_meta, val_meta = train_test_split(
        harp_meta,
        test_size=num_of_val,
        random_state=random_seed,
        shuffle=True,
        stratify=harp_meta['Group']
    )
    return list(train_meta['Subject']), list(val_meta['Subject'])


class HarpDataset(Dataset):

    def __init__(
            self,
            dir_path,
            brain_side,
            transforms=None
    ):
        """
        :param dir_path: Path to directory containing the subdirectories of each subject
        :param transforms: List of Transforms object (take note, NOT transforms.Compose object)
        Reason for using list of Transforms is to enable us to skip intensity transforms for hippocampus masks.
        """
        self.dir_path = dir_path
        if not isinstance(brain_side, str):
            raise TypeError("brain_side must be a string of either 'L' or 'R'")
        self.brain_side = brain_side
        self.transforms = transforms

    def __len__(self):
        num = 0
        for i in os.listdir(self.dir_path):
            if not i.startswith('.'):
                num += 1
        return num

    def __getitem__(
            self,
            subj_id
    ):
        """
        Lazily loads each subject's MRI volume and hippocampus segmentation masks only when required.
        :param subj_id: Subject ID of the patient
        :return: MRI volume and hippocampus segmentation labels as PyTorch Tensors
        """
        if str.upper(self.brain_side) == "L":
            vol_path = os.path.join(self.dir_path, subj_id, "SMALL_{0}_LB.nii".format(subj_id))
            label_path = os.path.join(self.dir_path, subj_id, "SMALL_{0}_LH.nii".format(subj_id))
        elif str.upper(self.brain_side) == "R":
            vol_path = os.path.join(self.dir_path, subj_id, "SMALL_{0}_RB.nii".format(subj_id))
            label_path = os.path.join(self.dir_path, subj_id, "SMALL_{0}_RH.nii".format(subj_id))

        # Use expand_dims to add one more singleton dimensions for the channel (PyTorch requires 4D tensors)
        mri_vol = np.expand_dims(np.asarray(nib.load(vol_path).dataobj), 0)
        hip_label = np.expand_dims(np.asarray(nib.load(label_path).dataobj), 0)

        mri_vol = mri_vol.astype(np.float32, casting='same_kind')
        hip_label = hip_label.astype(np.float32, casting='same_kind')

        mri_tensor = torch.from_numpy(mri_vol)
        hip_tensor = torch.from_numpy(hip_label)

        if self.transforms:
            for transform in self.transforms:
                tmp_mri = tio.Subject(default_image_name=tio.ScalarImage(tensor=mri_tensor, type=tio.INTENSITY))
                tmp_mri = transform(tmp_mri)
                fixed_transform = tmp_mri.get_applied_transforms()[0]
                mri_tensor = tmp_mri['default_image_name'].tensor
                if not isinstance(fixed_transform, IntensityTransform):
                    tmp_hip = tio.Subject(default_image_name=tio.LabelMap(tensor=hip_tensor, type=tio.LABEL))
                    tmp_hip = fixed_transform(tmp_hip)
                    hip_tensor = tmp_hip['default_image_name'].tensor

        return mri_tensor, hip_tensor

