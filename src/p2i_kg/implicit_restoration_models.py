"""Small untrained examples of coordinate, operator and restoration networks."""
import math
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, inputs, outputs, first=False, omega=30.):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.omega = omega
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / omega
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class TinySIREN(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = SineLayer(2, 16, first=True)
        self.hidden = SineLayer(16, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        return self.output(self.hidden(self.first(x)))


class TinyDeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 8))
        self.trunk = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 8))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, samples, coordinate):
        return (self.branch(samples) * self.trunk(coordinate)).sum(-1, keepdim=True) + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class TinyNAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.gate = SimpleGate()
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.gate(self.depthwise(self.expand(self.norm(x))))
        return x + self.project(y * self.channel_attention(y)) * self.scale


class TinyNAFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(3, 8, 3, padding=1)
        self.block = TinyNAFBlock(8)
        self.ending = nn.Conv2d(8, 3, 3, padding=1)

    def forward(self, x):
        return x + self.ending(self.block(self.intro(x)))
