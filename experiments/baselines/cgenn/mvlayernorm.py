# from https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks
import torch
from torch import nn

from experiments.baselines.cgenn.utils import unsqueeze_like

EPS = 1e-6


class MVLayerNorm(nn.Module):
    def __init__(self, algebra, channels):
        super().__init__()
        self.algebra = algebra
        self.channels = channels
        self.a = nn.Parameter(torch.ones(1, channels))

    def forward(self, input):
        norm = self.algebra.norm(input)[..., :1].mean(dim=1, keepdim=True) + EPS
        a = unsqueeze_like(self.a, norm, dim=2)
        return a * input / norm
