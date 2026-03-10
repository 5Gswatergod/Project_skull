from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def latest_checkpoint_path(run_dir: str | Path) -> Optional[Path]:
    run_dir = Path(run_dir)
    latest = run_dir / "latest.pt"
    if latest.exists():
        return latest

    ckpts = sorted(run_dir.glob("step_*.pt"))
    if ckpts:
        return ckpts[-1]
    return None


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    step: int = 0,
    best_val_loss: float | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model": model.state_dict(),
        "step": int(step),
        "best_val_loss": best_val_loss,
        "extra_state": extra_state or {},
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict:
    state = torch.load(path, map_location=map_location)

    model.load_state_dict(state["model"], strict=strict)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    return state
