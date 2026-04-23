from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import torch


_STEP_CHECKPOINT_RE = re.compile(r"^(?:interrupt_)?step_(\d+)\.pt$")


def _checkpoint_step(path: Path) -> Optional[int]:
    match = _STEP_CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _checkpoint_payload_step(path: Path) -> Optional[int]:
    try:
        state = torch.load(path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(state, dict):
        return None

    step = state.get("step")
    if step is None:
        return None

    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def latest_checkpoint_path(run_dir: str | Path) -> Optional[Path]:
    run_dir = Path(run_dir)
    latest = run_dir / "latest.pt"

    candidates = []
    for path in set(run_dir.glob("step_*.pt")) | set(run_dir.glob("interrupt_step_*.pt")):
        step = _checkpoint_step(path)
        if step is None:
            continue
        candidates.append((step, path.stat().st_mtime, 0, path))

    if latest.exists():
        latest_step = _checkpoint_payload_step(latest)
        if latest_step is not None:
            candidates.append((latest_step, latest.stat().st_mtime, 1, latest))
        elif not candidates:
            return latest

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[-1][3]
    return None


def resolve_checkpoint_path(path: str | Path) -> Optional[Path]:
    path = Path(path)
    if path.is_dir():
        return latest_checkpoint_path(path)
    if path.exists():
        return path
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

    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


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
