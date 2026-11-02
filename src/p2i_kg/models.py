"""Torch-dependent teaching models, imported only for runtime use."""
from torch import nn


class ThresholdPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return (self.linear(x) >= 0).to(x.dtype)


class TinyMixer(nn.Module):
    """Small structural example of Mixer; no attention or pretrained weights."""
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 16, kernel_size=4, stride=4)
        self.token_norm = nn.LayerNorm(16)
        self.token_mixer = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 16))
        self.channel_norm = nn.LayerNorm(16)
        self.channel_mixer = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))
        self.norm = nn.LayerNorm(16)
        self.head = nn.Linear(16, 4)

    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mixer(self.channel_norm(x))
        return self.head(self.norm(x).mean(1))


class TinyMAE(nn.Module):
    """Fixed-mask MAE example: visible-only encoder, unshuffle, masked MSE.

    The fixed permutation makes inspection and checkpoint comparisons repeatable.
    Inputs include all image pixels, but masked pixel embeddings never enter the
    encoder. Positions are small fixed buffers, not the official sin-cos values.
    """
    def __init__(self):
        super().__init__()
        import torch
        self.patch_embed = nn.Conv2d(3, 16, kernel_size=4, stride=4)
        self.encoder = nn.TransformerEncoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.decoder_embed = nn.Linear(16, 8)
        self.decoder = nn.TransformerEncoderLayer(8, 2, 16, dropout=0., batch_first=True)
        self.head = nn.Linear(8, 48)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 8))
        order = torch.tensor([0, 5, 10, 15, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
        self.register_buffer('visible', order[:4])
        self.register_buffer('restore', order.argsort())
        self.register_buffer('masked', order[4:])
        self.register_buffer('encoder_position', torch.arange(256).reshape(1, 16, 16).float() / 256)
        self.register_buffer('decoder_position', torch.arange(128).reshape(1, 16, 8).float() / 128)

    def forward(self, images):
        import torch
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        visible = (tokens + self.encoder_position).index_select(1, self.visible)
        latent = self.decoder_embed(self.encoder(visible))
        shuffled = torch.cat((latent, self.mask_token.expand(images.shape[0], 12, -1)), dim=1)
        restored = shuffled.index_select(1, self.restore)
        return self.head(self.decoder(restored + self.decoder_position))

    def reconstruction_loss(self, images, predictions):
        # Same channel-last ordering as the official patchify/einsum path.
        targets = images.reshape(images.shape[0], 3, 4, 4, 4, 4)
        targets = targets.permute(0, 2, 4, 3, 5, 1).reshape(images.shape[0], 16, 48)
        return (predictions - targets).square().mean(-1).index_select(1, self.masked).mean()
