from dataset import *


def get_dataset(cfg):
    if cfg.dataset.name == 'MRNet':
        return MRNetDataset(root_dir=cfg.dataset.root_dir, task=cfg.dataset.task, plane=cfg.dataset.plane, split=cfg.dataset.split)
    if cfg.dataset.name == 'BRATS':
        return BRATSDataset(root_dir=cfg.dataset.root_dir, train=cfg.dataset.train, img_type=cfg.dataset.img_type)
    if cfg.dataset.name == 'ADNI':
        return ADNIDataset()
    raise ValueError(f'{cfg.dataset.name} Dataset is not available')
