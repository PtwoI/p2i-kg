"""Reduced image classifiers exposing three different token-mixing operators."""
import torch
from torch import nn


class PoolMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, width)
        self.mixer = PoolMix()
        self.norm2 = nn.GroupNorm(1, width)
        self.channel_mlp = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.GELU(),
                                         nn.Conv2d(2 * width, width, 1))
        self.scale1 = nn.Parameter(torch.full((1, width, 1, 1), 1e-5))
        self.scale2 = nn.Parameter(torch.full((1, width, 1, 1), 1e-5))

    def forward(self, x):
        x = x + self.scale1 * self.mixer(self.norm1(x))
        return x + self.scale2 * self.channel_mlp(self.norm2(x))


class TinyPoolFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.block = PoolFormerBlock()
        self.norm = nn.GroupNorm(1, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, image):
        x = self.norm(self.block(self.patch_embed(image)))
        return self.head(x.mean(dim=(2, 3)))


class ConvMixerBlock(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width),
                                      nn.GELU(), nn.BatchNorm2d(width))
        self.channel = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                      nn.BatchNorm2d(width))

    def forward(self, x):
        return self.channel(x + self.spatial(x))


class TinyConvMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Sequential(nn.Conv2d(3, 8, 2, stride=2), nn.GELU(),
                                          nn.BatchNorm2d(8))
        self.block = ConvMixerBlock()
        self.head = nn.Linear(8, 3)

    def forward(self, image):
        x = self.block(self.patch_embed(image))
        return self.head(x.mean(dim=(2, 3)))


class Affine(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return x * self.scale + self.bias


class ResMLPBlock(nn.Module):
    def __init__(self, width=8, tokens=16):
        super().__init__()
        self.affine1 = Affine(width)
        self.token_mix = nn.Linear(tokens, tokens)
        self.affine2 = Affine(width)
        self.channel_mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                         nn.Linear(2 * width, width))
        self.scale1 = nn.Parameter(torch.full((width,), 1e-4))
        self.scale2 = nn.Parameter(torch.full((width,), 1e-4))

    def forward(self, x):
        mixed = self.token_mix(self.affine1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.scale1 * mixed
        return x + self.scale2 * self.channel_mlp(self.affine2(x))


class TinyResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 2, stride=2)
        self.block = ResMLPBlock()
        self.affine = Affine()
        self.head = nn.Linear(8, 3)

    def forward(self, image):
        patches = self.patch_embed(image).flatten(2).transpose(1, 2)
        return self.head(self.affine(self.block(patches)).mean(dim=1))
