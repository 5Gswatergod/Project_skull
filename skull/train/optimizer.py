from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    logits: [B, T, V]
    labels: [B, T]
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D [B, T, V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D [B, T], got {tuple(labels.shape)}")

    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
    return loss


@torch.no_grad()
def masked_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    preds = logits.argmax(dim=-1)
    mask = labels.ne(ignore_index)
    total = int(mask.sum().item())
    if total == 0:
        return 0.0
    correct = int(((preds == labels) & mask).sum().item())
    return correct / total
