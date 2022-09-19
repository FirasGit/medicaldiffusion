#!/bin/bash

source "/data/home/firas/anaconda3/etc/profile.d/conda.sh"
conda activate vq_gan_3d
export PYTHONPATH=$PWD

python train/train_ddpm.py model=ddpm dataset=brats model.results_folder_postfix='flair' model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/BRATS/flair/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=1