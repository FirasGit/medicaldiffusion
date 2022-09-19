from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse


class BreastUKA(Dataset):
    def __init__(self, path: str, split: str, preprocessing: Optional[tio.Compose] = None,
                 transforms: Optional[tio.Compose] = None):
        super().__init__()
        self.path = path
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.file_paths = self.get_data_files(split)

    def get_data_files(self, split: str):
        if split == 'train':
            return os.listdir(self.path)[:7]
        if split == 'val':
            return os.listdir(self.path)[7:]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = tio.ScalarImage(os.path.join(self.path, self.file_paths[idx]))
        return {'data': img.data.permute(0, -1, 1, 2)}

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--data_path', type=str,
                            default='/media/NAS/datasets/breast/data_firas')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--image_channels', type=int, default=1)
        return parser
