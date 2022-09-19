from re import I
from ddpm import Unet3D, GaussianDiffusion, Trainer
from dataset import MRNetDataset, BRATSDataset
import argparse
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from train.dataset import get_dataset


# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py

@hydra.main(config_path='../config', config_name='base_cfg')
def run(cfg: DictConfig):
    model = Unet3D(
        dim=cfg.model.unet.diffusion_img_size,
        dim_mults=cfg.model.unet.dim_mults,
        channels=cfg.model.unet.diffusion_num_channels,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).cuda()

    train_dataset, *_ = get_dataset(cfg)

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        # logger=cfg.model.logger
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    trainer.train()


if __name__ == '__main__':
    run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
