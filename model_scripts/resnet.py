from __future__ import annotations
from math import sqrt
import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from einops import reduce, rearrange
from einops.layers.torch import Rearrange
from data_scripts.helper_funcs import default, exists


class FiLM(Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.to_gamma = nn.Linear(dim, dim_out, bias=False)
        self.to_beta = nn.Linear(dim, dim_out)

        self.gamma_mult = nn.Parameter(torch.zeros(1,))
        self.beta_mult = nn.Parameter(torch.zeros(1,))

    def forward(self, x, cond):
        g, b = self.to_gamma(cond), self.to_beta(cond)
        g, b = tuple(rearrange(t, 'b d -> b 1 d') for t in (g, b))

        g = (1 + self.gamma_mult * g.tanh())
        b = b.tanh() * self.beta_mult

        return x * g + b


class PixelNorm(Module):
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim=self.dim, eps=self.eps) * sqrt(x.shape[self.dim])


class SqueezeExcite(Module):
    def __init__(self, dim, reduction_factor=4, min_dim=16):
        super().__init__()
        dim_inner = max(dim // reduction_factor, min_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Linear(dim_inner, dim),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1')
        )

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)
            num = reduce(x, 'b c n -> b c', 'sum')
            den = reduce(mask.float(), 'b 1 n -> b 1', 'sum')
            avg = num / den.clamp(min=1e-5)
        else:
            avg = reduce(x, 'b c n -> b c', 'mean')

        return x * self.net(avg)


class Block(Module):
    def __init__(self, dim, dim_out=None, dropout=0.):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = PixelNorm(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, mask=None):
        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.proj(x)

        if exists(mask):
            x = x.masked_fill(~mask, 0.)

        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        return x


class ResnetBlock(Module):
    def __init__(self, dim, dim_out=None, *, dropout=0.):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.excite = SqueezeExcite(dim_out)
        self.residual_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, mask=None):
        r = self.residual_conv(x)
        h = self.block1(x, mask=mask)
        h = self.block2(h, mask=mask)
        h = self.excite(h, mask=mask)
        return h + r