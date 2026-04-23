from __future__ import annotations

import os
from typing import Any

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - exercised via unit tests
    Accelerator = None


def accelerate_requested(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("use_accelerate", False))


def normalize_mixed_precision(value: Any) -> str:
    name = str(value or "no").lower()
    if name in {"fp16", "bf16"}:
        return name
    return "no"


def build_accelerator(cfg: dict[str, Any]):
    if not accelerate_requested(cfg):
        return None

    if Accelerator is None:
        raise ImportError(
            "Accelerate support was requested, but `accelerate` is not installed. "
            "Install it with `pip install -e .[accelerate]` or `pip install accelerate`."
        )

    requested_device = str(cfg.get("device", "") or "").lower()
    return Accelerator(
        cpu=requested_device == "cpu",
        mixed_precision=normalize_mixed_precision(cfg.get("mixed_precision", "no")),
        gradient_accumulation_steps=max(1, int(cfg.get("grad_accum", 1))),
    )


def is_primary_process_from_env() -> bool:
    for env_name in ("ACCELERATE_PROCESS_INDEX", "RANK", "LOCAL_RANK"):
        value = os.environ.get(env_name)
        if value is None:
            continue
        try:
            return int(value) == 0
        except ValueError:
            return value == "0"
    return True
