import math
import torch
import torch.nn as nn

from monai.networks.layers.utils import get_act_layer


class SinusoidalPosEmb(nn.Module):
    def __init__(self, emb_dim=16, downscale_freq_shift=1, max_period=10000, flip_sin_to_cos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, x):
        device = x.device
        half_dim = self.emb_dim // 2
        emb = math.log(self.max_period) / \
            (half_dim - self.downscale_freq_shift)
        emb = torch.exp(-emb*torch.arange(half_dim, device=device))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        half_dim = emb_dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        if self.emb_dim % 2 == 1:
            fouriered = torch.nn.functional.pad(fouriered, (0, 1, 0, 0))
        return fouriered


class TimeEmbbeding(nn.Module):
    def __init__(
        self,
        emb_dim=64,
        pos_embedder=SinusoidalPosEmb,
        pos_embedder_kwargs={},
        act_name=("SWISH", {})  # Swish = SiLU
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.pos_emb_dim = pos_embedder_kwargs.get('emb_dim', emb_dim//4)
        pos_embedder_kwargs['emb_dim'] = self.pos_emb_dim
        self.pos_embedder = pos_embedder(**pos_embedder_kwargs)

        self.time_emb = nn.Sequential(
            self.pos_embedder,
            nn.Linear(self.pos_emb_dim, self.emb_dim),
            get_act_layer(act_name),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

    def forward(self, time):
        return self.time_emb(time)
