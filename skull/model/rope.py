from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(start_dim=-2)


def precompute_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device=None,
    dtype=torch.float32,
):
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)
    return cos, sin


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    q, k: [B, nh, T, hs]
    cos, sin: [T, hs]
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,hs]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
