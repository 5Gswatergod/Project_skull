from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden = max(1, int(config.n_embd * float(config.mlp_hidden_mult)))
        self.fc = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        x = self.dropout(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden = max(1, int(config.n_embd * float(config.mlp_hidden_mult)))
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w_gate(x)) * self.w_up(x)
        x = self.w_down(x)
        x = self.dropout(x)
        return x


def build_mlp(config) -> nn.Module:
    mlp_type = str(getattr(config, "mlp_type", "gelu")).lower()
    if mlp_type == "gelu":
        return MLP(config)
    if mlp_type == "swiglu":
        return SwiGLUMLP(config)
    raise ValueError(f"Unsupported mlp_type: {mlp_type}")
