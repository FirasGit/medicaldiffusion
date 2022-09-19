import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from vq_gan_3d.model import VQGAN
from dataset import MRNetDataset, BRATSDataset, ADNIDataset
from train.callbacks import ImageLogger, VideoLogger

# TO TRAIN:
# export PYTHONPATH=$PWD in previous folder
# NCCL_DEBUG=WARN PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py --gpus 1 --default_root_dir /data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints/knee_mri --precision 16 --embedding_dim 256 --n_hiddens 16 --downsample 16 16 16 --num_workers 32 --gradient_clip_val 1.0 --lr 3e-4 --discriminator_iter_start 10000 --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 --batch_size 2 --n_codes 1024 --accumulate_grad_batches 1
# PL_TORCH_DISTRIBUTED_BACKEND=gloo CUDA_VISIBLE_DEVICES=1 python train/train_vqgan.py --gpus 1 --default_root_dir /data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints_generation/knee_mri_gen --precision 16 --embedding_dim 8 --n_hiddens 16 --downsample 8 8 8 --num_workers 32 --gradient_clip_val 1.0 --lr 3e-4 --discriminator_iter_start 10000 --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 --batch_size 2 --n_codes 16384 --accumulate_grad_batches 1
# https://github.com/Lightning-AI/lightning/issues/9641

# PL_TORCH_DISTRIBUTED_BACKEND=gloo CUDA_VISIBLE_DEVICES=1 python train/train_vqgan.py --gpus 1 --default_root_dir /data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints_brats/flair --precision 16 --embedding_dim 8 --n_hiddens 16 --downsample 2 2 2 --num_workers 32 --gradient_clip_val 1.0 --lr 3e-4 --discriminator_iter_start 10000 --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1 --gan_feat_weight 4 --batch_size 2 --n_codes 16384 --accumulate_grad_batches 1 --dataset BRATS


def main():
    DATASET = BRATSDataset

    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQGAN.add_model_specific_args(parser)
    parser = DATASET.add_data_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == 'MRNet':
        train_dataset = MRNetDataset(
            root_dir=args.root_dir, task=args.task, plane=args.plane, split='train')
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
        val_dataset = MRNetDataset(
            root_dir=args.root_dir, task=args.task, plane=args.plane, split='valid')
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'BRATS':
        train_dataset = BRATSDataset(
            root_dir=args.root_dir, imgtype=args.imgtype, train=True)
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        val_dataset = BRATSDataset(
            root_dir=args.root_dir, imgtype=args.imgtype, train=False)
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    elif args.dataset == 'ADNI':
        train_dataset = MRNetDataset(
            root_dir=args.root_dir, task=args.task, plane=args.plane, split='train')
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
        val_dataset = MRNetDataset(
            root_dir=args.root_dir, task=args.task, plane=args.plane, split='valid')
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = args.batch_size, args.lr, args.gpus, args.accumulate_grad_batches
    args.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        args.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(accelerator='ddp', gpus=args.gpus)

    # load the most recent checkpoint file
    base_dir = os.path.join(args.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                args.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      args.resume_from_checkpoint)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps, **kwargs)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()
