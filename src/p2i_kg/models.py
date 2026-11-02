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


class TinyDETR(nn.Module):
    """Small DETR-shaped detector with learned queries and parallel set output."""
    def __init__(self):
        super().__init__()
        import torch
        self.backbone = nn.Conv2d(3, 16, kernel_size=4, stride=4)
        self.encoder = nn.TransformerEncoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.decoder = nn.TransformerDecoderLayer(16, 2, 32, dropout=0., batch_first=True)
        self.query_embed = nn.Embedding(5, 16)
        self.class_head = nn.Linear(16, 4)  # three classes plus no-object
        self.box_head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        self.register_buffer('position', torch.arange(256).reshape(1, 16, 16).float() / 256)

    def forward(self, images):
        tokens = self.backbone(images).flatten(2).transpose(1, 2) + self.position
        memory = self.encoder(tokens)
        queries = self.query_embed.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return __import__('torch').cat((self.class_head(decoded), self.box_head(decoded).sigmoid()), -1)

    def set_loss(self, predictions, target_classes, target_boxes):
        """Exact tiny bipartite assignment for two targets, then class/L1 loss."""
        import itertools
        import torch
        from torch.nn import functional as F
        assignments = list(itertools.permutations(range(predictions.shape[1]), 2))
        batch_losses, chosen_assignments = [], []
        for batch, (classes, boxes) in enumerate(zip(target_classes, target_boxes)):
            losses = []
            for first, second in assignments:
                chosen = predictions[batch, (first, second)]
                losses.append(F.cross_entropy(chosen[:, :4], classes) +
                    F.l1_loss(chosen[:, 4:], boxes))
            stacked = torch.stack(losses)
            index = int(stacked.detach().argmin())
            batch_losses.append(stacked[index])
            chosen_assignments.append(assignments[index])
        return torch.stack(batch_losses).mean(), chosen_assignments


class FourierPositionEncoder(nn.Module):
    """NeRF-style identity plus log-spaced sine/cosine features."""
    def __init__(self, frequencies):
        super().__init__()
        import torch
        self.register_buffer('frequencies', 2. ** torch.arange(frequencies).float())

    def forward(self, values):
        import torch
        encoded = [values]
        for frequency in self.frequencies:
            encoded.extend((torch.sin(values * frequency), torch.cos(values * frequency)))
        return torch.cat(encoded, -1)


class VolumeRenderer(nn.Module):
    """Alpha compositing along one ordered sample sequence per ray."""
    def forward(self, rgb, density, deltas):
        import torch
        alpha = 1. - torch.exp(-torch.relu(density.squeeze(-1)) * deltas)
        transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-6), 1), 1)[:, :-1]
        weights = alpha * transmittance
        return (weights.unsqueeze(-1) * rgb).sum(1)


class TinyNeRF(nn.Module):
    """Small NeRF-shaped field with view-dependent color and volume rendering."""
    def __init__(self):
        super().__init__()
        self.position_encoder = FourierPositionEncoder(2)
        self.direction_encoder = FourierPositionEncoder(1)
        self.point_mlp = nn.Sequential(nn.Linear(15, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.density_head = nn.Linear(32, 1)
        self.feature_head = nn.Linear(32, 16)
        self.view_head = nn.Sequential(nn.Linear(25, 16), nn.ReLU(), nn.Linear(16, 3), nn.Sigmoid())
        self.volume_renderer = VolumeRenderer()

    def forward(self, points, directions, deltas):
        features = self.point_mlp(self.position_encoder(points))
        density = self.density_head(features)
        view_input = __import__('torch').cat((self.feature_head(features), self.direction_encoder(directions)), -1)
        rgb = self.view_head(view_input)
        return self.volume_renderer(rgb, density, deltas)
