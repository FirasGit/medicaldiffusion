#!/bin/bash

source "/data/home/firas/anaconda3/etc/profile.d/conda.sh"
conda activate vq_gan_3d
export PYTHONPATH=$PWD

# MRNet
python train/train_ddpm.py model=ddpm dataset=mrnet model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/MRNet/lightning_logs/version_0/checkpoints/epoch\=126-step\=114000-train/recon_loss\=0.47.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=4 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=40 model.gpus=1 model.load_milestone='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/MRNet/model-8.pt'

# ADNI
#python train/train_ddpm.py model=ddpm dataset=adni model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/ADNI/lightning_logs/version_0/checkpoints/epoch\=568-step\=567000-train/recon_loss\=0.02.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=2 model.load_milestone='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/ADNI/model-19.pt'
python train/train_ddpm.py model=ddpm dataset=adni model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/ADNI/roi/lightning_logs/version_1/checkpoints/epoch\=99-step\=99000-train/recon_loss\=0.05.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=2 model.results_folder_postfix='roi'
python train/train_ddpm.py model=ddpm dataset=adni model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/ADNI/roi/lightning_logs/version_1/checkpoints/epoch\=99-step\=99000-train/recon_loss\=0.05.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=3 model.results_folder_postfix='roi_normal_unet' model.denoising_fn='UNet'

# DUKE
#python train/train_ddpm.py model=ddpm dataset=duke model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/DUKE/lightning_logs/version_0/checkpoints/epoch\=58-step\=108000-train/recon_loss\=0.17.ckpt' model.diffusion_img_size=32 model.diffusion_depth_size=4 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=40 model.gpus=3 model.load_milestone='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/DUKE/model-10.pt'
python train/train_ddpm.py model=ddpm dataset=duke model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/DUKE/low_compression/lightning_logs/version_0/checkpoints/epoch\=69-step\=129000-train/recon_loss\=0.11.ckpt' model.diffusion_img_size=64 model.diffusion_depth_size=8 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=0 model.results_folder_postfix='low_compression'

# LIDC
python train/train_ddpm.py model=ddpm dataset=lidc model.vqgan_ckpt='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/LIDC/lightning_logs/version_0/checkpoints/epoch\=100-step\=102000-train/recon_loss\=0.33.ckpt' model.diffusion_img_size=16 model.diffusion_depth_size=16 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=50 model.gpus=4 model.load_milestone='/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/LIDC/model-20.pt'