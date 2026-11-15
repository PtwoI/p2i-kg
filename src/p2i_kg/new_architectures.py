"""Small, randomly initialized architectural examples, not trained replicas."""
import torch
from torch import nn
import torch.nn.functional as F

class SplineEdges(nn.Module):
    """Degree-one B-spline edges on a fixed grid, summed over input channels."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.register_buffer('knots', torch.linspace(-2, 2, 9))
        self.coefficients = nn.Parameter(torch.randn(outputs, inputs, 9) * .1)
        self.base = nn.Linear(inputs, outputs, bias=False)
    def forward(self, x):
        basis = (1 - (x.unsqueeze(-1) - self.knots).abs() / .5).clamp(min=0)
        return torch.einsum('...ik,oik->...o', basis, self.coefficients) + self.base(F.silu(x))

class TinyKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.spline_one = SplineEdges(3, 5)
        self.spline_two = SplineEdges(5, 1)
    def forward(self, x):
        return self.spline_two(self.spline_one(x))

class ScalarMemory(nn.Module):
    """sLSTM scalar memory with exponential input/forget gates and stabilization."""
    def __init__(self, width):
        super().__init__()
        self.gates = nn.Linear(width * 2, width * 4)
    def forward(self, x):
        h = torch.zeros_like(x[:, 0]); c = torch.zeros_like(h); n = torch.zeros_like(h); m = torch.zeros_like(h)
        outputs = []
        for t in range(x.shape[1]):
            i, f, z, o = self.gates(torch.cat((x[:, t], h), dim=-1)).chunk(4, -1)
            m_new = torch.maximum(f + m, i)
            it = torch.exp(i - m_new); ft = torch.exp(f + m - m_new)
            c = ft * c + it * torch.tanh(z); n = ft * n + it
            h = torch.sigmoid(o) * c / n.clamp_min(1e-6); m = m_new
            outputs.append(h)
        return torch.stack(outputs, dim=1)

class TinyXLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(4, 8)
        self.memory = ScalarMemory(8)
        self.output_projection = nn.Linear(8, 4)
    def forward(self, x):
        return self.output_projection(self.memory(self.input_projection(x)))

class Decompose(nn.Module):
    def forward(self, x):
        trend = F.avg_pool1d(F.pad(x.unsqueeze(1), (1, 1), mode='replicate'), 3, stride=1).squeeze(1)
        return x - trend, trend

class TinyTimeMixer(nn.Module):
    """Two-scale seasonal and trend mixing and multiscale prediction."""
    def __init__(self):
        super().__init__()
        self.decomposition = Decompose()
        self.season_down = nn.Linear(8, 4)
        self.trend_up = nn.Linear(4, 8)
        self.predict_fine = nn.Linear(8, 3)
        self.predict_coarse = nn.Linear(4, 3)
    def forward(self, x):
        coarse = F.avg_pool1d(x.unsqueeze(1), 2, stride=2).squeeze(1)
        s0, t0 = self.decomposition(x); s1, t1 = self.decomposition(coarse)
        s1 = s1 + self.season_down(s0)
        t0 = t0 + self.trend_up(t1)
        return self.predict_fine(s0 + t0) + self.predict_coarse(s1 + t1)
