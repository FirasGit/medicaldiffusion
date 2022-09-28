import os
import torchio as tio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import signal
import random
import math
import argparse

# TODO: Normalize to -1 and 1


def reformat_label(label):
    if label == 1:
        label = torch.FloatTensor([1])
    elif label == 0:
        label = torch.FloatTensor([0])
    return label


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(256, 256, 32))
])

TRAIN_TRANSFORMS = tio.Compose([
    # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
    # 0, 0, 3), translation=(4, 4, 0)),
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

VAL_TRANSFORMS = None


class MRNetDataset(Dataset):
    def __init__(self, root_dir, task, plane, split='train', preprocessing_transforms=None, transforms=None, fold=0):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.split = split
        self.preprocessing_transforms = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS if split == 'train' else VAL_TRANSFORMS
        self.fold = fold

        # Load labels
        self.records = self._get_annotations()
        self.records['id'] = self._remap_id_to_match_folder_name()
        self.labels = self.records['label'].tolist()

        self.paths = self._get_file_paths()

    def _get_file_paths(self):
        path_split = self.split if self.split == 'test' else 'train'
        file_paths = []
        for filename in self.records['id'].tolist():
            plane_paths = {}
            for plane in ['axial', 'coronal', 'sagittal']:
                plane_paths[plane] = self.root_dir + \
                    '{0}/{1}/'.format(path_split, plane) + \
                    filename + '.npy'
            file_paths.append(plane_paths)
        return file_paths

    def _remap_id_to_match_folder_name(self):
        return self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))

    def _get_annotations(self):
        path_split = self.split if self.split == 'test' else 'train'
        records = pd.read_csv(
            self.root_dir + '{0}-{1}.csv'.format(path_split, self.task),
            header=None, names=['id', 'label'])

        if self.fold != None:
            indexes = list(range(0, 1130))
            random.seed(26)
            random.shuffle(indexes)
            num_folds = 5
            ind = math.floor(len(indexes) / num_folds)
            if self.fold == num_folds - 1:
                valid_ind = indexes[ind*(self.fold):]
                train_ind = np.setdiff1d(indexes, valid_ind)
            else:
                valid_ind = indexes[ind*(self.fold):ind*(self.fold+1)]
                train_ind = np.setdiff1d(indexes, valid_ind)
            if self.split == 'train':
                records = records[records['id'].isin(train_ind)]
            if self.split == 'valid':
                records = records[records['id'].isin(valid_ind)]

        return records

    def __len__(self):
        return len(self.paths)

    @property
    def sample_weight(self):
        class_sample_count = np.unique(self.labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[self.labels]
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def __getitem__(self, index):
        # Load image data and label
        array = {}
        array_org = {}
        for plane in self.paths[index]:
            if self.plane != 'all' and plane != self.plane:
                continue
            _array = np.load(self.paths[index][plane])
            _array = _array.astype('float32')
            _array = _array[None]  # Add channel dimension
            array_org[plane] = _array
            _array = _array.transpose(0, 2, 3, 1)  # Use C, H, W, D for TorchIO
            if self.preprocessing_transforms:
                _array = self.preprocessing_transforms(_array)
            if self.transforms:
                _array = self.transforms(_array)
            # Revert to C, D, H, W for PyTorch
            _array = _array.transpose(0, 3, 1, 2)
            if plane != 'axial':  # Praxis dataset denotes axial as transversal. Make sure it's the same
                array[plane] = _array
            else:
                array['transversal'] = _array
        label = self.labels[index]
        label = reformat_label(label)

        # Sample identifier
        id = self.paths[index][plane].split('/')[-1].split('.')[0]

        data = array[self.plane]
        return {'data': data}
