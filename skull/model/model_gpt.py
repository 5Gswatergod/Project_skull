from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope, precompute_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.bias = config.bias

        self.pos_encoding = str(config.pos_encoding).lower()
        self.rope_base = float(config.rope_base)
        self.block_size = int(config.block_size)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = hasattr(F, "scaled_dot_product_attention")

        if not self.use_flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "bias_mask",
                mask.view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )
        else:
            self.register_buffer("bias_mask", None, persistent=False)

        if self.pos_encoding == "rope":
            cos, sin = precompute_rope_cache(
                seq_len=config.block_size,
                dim=self.head_dim,
                base=self.rope_base,
                device=None,
                dtype=torch.float32,
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)

    def _get_rope_cache(self, t: int, device, dtype):
        cos = self.rope_cos[:t].to(device=device, dtype=dtype)
        sin = self.rope_sin[:t].to(device=device, dtype=dtype)
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, nh, T, hs]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.pos_encoding == "rope":
            cos, sin = self._get_rope_cache(T, device=q.device, dtype=q.dtype)
            q, k = apply_rope(q, k, cos, sin)

        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y
