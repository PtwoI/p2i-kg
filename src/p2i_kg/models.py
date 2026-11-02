"""Torch-dependent teaching models, imported only for runtime use."""
from torch import nn


class ThresholdPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return (self.linear(x) >= 0).to(x.dtype)
