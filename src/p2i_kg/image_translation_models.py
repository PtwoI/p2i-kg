"""Small, randomly initialized examples of nested U and image translation models."""
import torch
from torch import nn


class ResidualUBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encode = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.decode = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, image):
        skip = self.encode(image).relu()
        deep = self.up(self.bottleneck(self.pool(skip)).relu())
        return image + self.decode(torch.cat((skip, deep), dim=1)).relu()


class TinyU2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.encoder = ResidualUBlock(4)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ResidualUBlock(4)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder = ResidualUBlock(8)
        self.side_deep = nn.Conv2d(4, 1, 1)
        self.side_shallow = nn.Conv2d(8, 1, 1)
        self.fusion = nn.Conv2d(2, 1, 1)

    def forward(self, image):
        skip = self.encoder(self.stem(image).relu())
        deep = self.bridge(self.pool(skip))
        decoded = self.decoder(torch.cat((skip, self.up(deep)), dim=1))
        side_deep = self.up(self.side_deep(deep))
        side_shallow = self.side_shallow(decoded)
        return self.fusion(torch.cat((side_shallow, side_deep), dim=1)), side_shallow, side_deep


class SkipGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.down = nn.Conv2d(4, 8, 4, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1)
        self.output = nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, source):
        skip = self.stem(source).relu()
        decoded = self.up(self.down(skip).relu()).relu()
        return self.output(torch.cat((skip, decoded), dim=1)).tanh()


def patch_discriminator(channels):
    return nn.Sequential(
        nn.Conv2d(channels, 4, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(4, 1, 3, padding=1),
    )


class TinyPix2Pix(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = SkipGenerator()
        self.discriminator = patch_discriminator(6)

    def forward(self, source, target):
        generated = self.generator(source)
        fake = self.discriminator(torch.cat((source, generated), dim=1))
        real = self.discriminator(torch.cat((source, target), dim=1))
        return generated, fake, real


class ResidualImageBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1)
        )

    def forward(self, x):
        return x + self.transform(x)


def residual_generator():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        ResidualImageBlock(), nn.Conv2d(4, 3, 3, padding=1), nn.Tanh(),
    )


class TinyCycleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_to_b = residual_generator()
        self.b_to_a = residual_generator()
        self.discriminator_a = patch_discriminator(3)
        self.discriminator_b = patch_discriminator(3)

    def forward(self, domain_a, domain_b):
        fake_b = self.a_to_b(domain_a)
        fake_a = self.b_to_a(domain_b)
        reconstructed_a = self.b_to_a(fake_b)
        reconstructed_b = self.a_to_b(fake_a)
        return (
            reconstructed_a, reconstructed_b,
            self.discriminator_a(fake_a), self.discriminator_b(fake_b),
            self.discriminator_a(domain_a), self.discriminator_b(domain_b),
        )
