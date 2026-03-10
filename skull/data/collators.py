from __future__ import annotations

from typing import Sequence

import torch


def causal_lm_collate_fn(batch: Sequence[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)

    out = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
    }

    if "source_id" in batch[0]:
        out["source_id"] = torch.stack([x["source_id"] for x in batch], dim=0)
    if "shard_id" in batch[0]:
        out["shard_id"] = torch.stack([x["shard_id"] for x in batch], dim=0)

    return out


def sft_collate_fn(batch: Sequence[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
