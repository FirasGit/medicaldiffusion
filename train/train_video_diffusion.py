from re import I
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
from dataset import MRNetDataset, BRATSDataset
import argparse
import wandb

# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str,
                    default='/data/home/firas/Desktop/work/MR_Knie/Data/MRNet/MRNet-v1.0/')
parser.add_argument('--vqgan_ckpt', type=str,
                    default='/data/home/firas/Desktop/work/other_groups/vq_gan_3d/checkpoints_generation/knee_mri_gen/lightning_logs/version_0/checkpoints/epoch=245-step=222000-train/recon_loss=0.81.ckpt')
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=30)
parser.add_argument('--image_channels', type=int, default=1)
parser.add_argument('--task', type=str, default='acl')
parser.add_argument('--plane', type=str, default='sagittal')
parser.add_argument('--load_milestone', type=str, default=None)
parser.add_argument('--logger', type=str, default='wandb')
parser.add_argument('--objective', type=str, default='pred_x0')
parser.add_argument('--diffusion_img_size', type=int, default=32)
parser.add_argument('--diffusion_depth_size', type=int, default=4)
parser.add_argument('--diffusion_num_channels', type=int, default=8)
parser.add_argument('--save_and_sample_every', type=int, default=400)
parser.add_argument('--train_lr', type=float, default=1e-4)
args = parser.parse_args()

# if args.logger == 'wandb':
# logger = wandb.init(entity='infinite_imaging',
# project='3d_diffusion')
# else:
#logger = None


model = Unet3D(
    dim=args.diffusion_img_size,
    dim_mults=(1, 2, 4, 8),
    channels=args.diffusion_num_channels,
).cuda()

diffusion = GaussianDiffusion(
    model,
    vqgan_ckpt=args.vqgan_ckpt,
    image_size=args.diffusion_img_size,
    num_frames=args.diffusion_depth_size,
    channels=args.diffusion_num_channels,
    timesteps=300,           # number of steps
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    # sampling_timesteps=250,
    loss_type='l1',            # L1 or L2
    # objective=args.objective
).cuda()

# train_dataset = MRNetDataset(
# root_dir=args.root_dir, task=args.task, plane=args.plane, split='train')

train_dataset = BRATSDataset(
    root_dir=args.root_dir, train=True, imgtype='flair')

trainer = Trainer(
    diffusion,
    args=args,
    dataset=train_dataset,
    train_batch_size=args.batch_size,
    save_and_sample_every=args.save_and_sample_every,
    train_lr=args.train_lr,
    train_num_steps=700000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                       # turn on mixed precision
    num_sample_rows=1
    # logger=logger
)

if args.load_milestone:
    trainer.load(args.load_milestone)

trainer.train()

# wandb.finish()

# Incorporate GAN loss in DDPM training?
# Incorporate GAN loss in UNET segmentation?
# Maybe better if I don't use ema updates?
# Use with other vqgan latent space (the one with more channels?)
