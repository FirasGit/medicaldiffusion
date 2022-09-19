import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
import argparse


class ADNIDataset(Dataset):
    def __init__(self, root='../ADNI', augmentation=False):
        self.root = root
        self.basis = 'FreeSurfer_Cross-Sectional_Processing_brainmask'
        self.augmentation = augmentation
        f = open('CN_list.csv', 'r')
        rdr = csv.reader(f)

        name = []
        labels = []
        date = []
        for line in rdr:
            [month, day, year] = line[9].split('/')
            month = month.zfill(2)
            date.append(year+'-'+month+'-'+day)
            name.append(line[1])

        name = np.asarray(name)
        date = np.asarray(date)

        self.name = name
        self.date = date

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.name[index], self.basis)
        files = os.listdir(path)
        for file in files:
            if file[:10] == self.date[index]:
                rname = file
        aname = os.listdir(os.path.join(path, rname))[0]
        path = os.path.join(path, rname, aname, 'mri')
        img = nib.load(os.path.join(path, 'image.nii'))

        img = np.swapaxes(img.get_data(), 1, 2)
        img = np.flip(img, 1)
        img = np.flip(img, 2)
        sp_size = 64
        img = resize(img, (sp_size, sp_size, sp_size), mode='constant')
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3*torch.rand(1)[0]+0.7
            if random_n[0] > 0.5:
                img = np.flip(img, 0)

            img = img*random_i.data.cpu().numpy()

        imageout = torch.from_numpy(img).float().view(
            1, sp_size, sp_size, sp_size)
        imageout = imageout*2-1

        return imageout

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--root_dir', type=str,
                            default='/data/home/firas/Desktop/work/MR_Knie/Data/MRNet/MRNet-v1.0/')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--image_channels', type=int, default=1)
        parser.add_argument('--task', type=str, default='acl')
        parser.add_argument('--plane', type=str, default='sagittal')
        return parser
