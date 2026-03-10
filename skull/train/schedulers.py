from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_lr_lambda(
    schedule_type: str,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    schedule_type = str(schedule_type).lower()
    warmup_steps = max(0, int(warmup_steps))
    max_steps = max(1, int(max_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        if schedule_type == "constant":
            return 1.0

        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if schedule_type == "linear":
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

        if schedule_type == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        raise ValueError(f"Unsupported schedule_type: {schedule_type}")

    return lr_lambda


def build_lr_scheduler(
    optimizer: Optimizer,
    cfg: dict,
) -> LambdaLR:
    schedule_type = cfg.get("lr_schedule", "cosine")
    warmup_steps = int(cfg.get("warmup_steps", 0))
    max_steps = int(cfg.get("max_steps", 1000))
    base_lr = float(cfg.get("lr", 3e-4))
    min_lr = float(cfg.get("min_lr", base_lr * 0.1))
    min_lr_ratio = min_lr / max(base_lr, 1e-12)

    return LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(
            schedule_type=schedule_type,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )
