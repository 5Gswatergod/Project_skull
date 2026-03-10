from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from skull.data import MultiBinDataset, causal_lm_collate_fn


@torch.no_grad()
def evaluate_perplexity_from_cfg(model, cfg: dict, device: torch.device) -> dict:
    eval_sources = cfg.get("eval_sources")
    if not eval_sources:
        raise ValueError("Eval config missing 'eval_sources'")

    block_size = int(cfg["block_size"])
    bin_dtype = str(cfg.get("bin_dtype", "uint32"))
    row_tokens = int(cfg.get("row_tokens", block_size + 1))
    batch_size = int(cfg.get("batch_size", 8))
    num_workers = int(cfg.get("num_workers", 0))
    nominal_size = int(cfg.get("eval_nominal_size", 20_000))
    eval_batches = int(cfg.get("eval_batches", 100))
    seed = int(cfg.get("seed", 42)) + 12345

    ds = MultiBinDataset(
        sources=eval_sources,
        block_size=block_size,
        dtype=bin_dtype,
        row_tokens=row_tokens,
        nominal_size=nominal_size,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=causal_lm_collate_fn,
    )

    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    for i, batch in enumerate(loader):
        if i >= eval_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        out = model(input_ids)
        logits = out["logits"] if isinstance(out, dict) else out
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)

        loss_sum = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="sum",
            ignore_index=-100,
        )
        valid_tokens = int(flat_labels.ne(-100).sum().item())

        total_loss += float(loss_sum.item())
        total_tokens += valid_tokens
        total_batches += 1

    if total_tokens == 0:
        raise RuntimeError("No valid tokens were evaluated.")

    return {
        "loss": total_loss / total_tokens,
        "tokens": total_tokens,
        "batches": total_batches,
    }
