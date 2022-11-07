# Medical Diffusion

This repository contains the code to our paper "Medical Diffusion: Denoising Diffusion Probabilistic Models for 3D Medical Image Synthesis"

![alt text](assets/main.png)

## Training
In order to run our model, we suggest you create a virtual environment ```conda create -n medicaldiffusion python=3.8``` and activate it with ```conda activate medicaldiffusion```. Subsequently, download and install the required libraries by running ```pip install -r requirements.txt```

Once all libraries are installed and the datasets have been downloaded, you are ready to train the model:
To train the 3D-VQ-GAN model on e.g. the BraTS dataset, you can run the following command

```
PL_TORCH_DISTRIBUTED_BACKEND=gloo python train/train_vqgan.py dataset=brats dataset.root_dir=<INSERT_PATH_TO_BRATS_DATASET> model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='flair' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=32 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=2 model.n_codes=16384 model.accumulate_grad_batches=1 
```
Note that you need to provide the path to the dataset (e.g. ```dataset.root_dir='/data/BraTS/BraTS 2020'```) to successfully run the command.

To train the diffusion model in the latent space of the previously trained VQ-GAN model, you need to run the following command
```
python train/train_ddpm.py model=ddpm dataset=brats model.results_folder_postfix='flair' model.vqgan_ckpt=<<INSERT_PATH_TO_CHECKPOINT> model.diffusion_img_size=32 model.diffusion_depth_size=32 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=10 model.gpus=1
```
Where you again need to specify the path to the VQ-GAN checkpoint from before (e.g. ```model.vqgan_ckpt='/home/<user>/Desktop/medicaldiffusion/checkpoints/vq_gan/BRATS/flair/lightning_logs/version_0/checkpoints/latest_checkpoint.ckpt'```)



## Acknowledgement
This code is heavily build on the following repositories:

(1) https://github.com/SongweiGe/TATS

(2) https://github.com/lucidrains/denoising-diffusion-pytorch

(3) https://github.com/lucidrains/video-diffusion-pytorch
