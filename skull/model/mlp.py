from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


def build_norm(norm_type: str, ndim: int, bias: bool = True, eps: float = 1e-5):
    norm_type = str(norm_type).lower()
    if norm_type == "layernorm":
        return LayerNorm(ndim, bias=bias, eps=eps)
    if norm_type == "rmsnorm":
        return RMSNorm(ndim, eps=eps)
    raise ValueError(f"Unsupported norm type: {norm_type}")
