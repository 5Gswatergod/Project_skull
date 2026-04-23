from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from skull.train.amp import build_grad_scaler
from skull.train.checkpointing import (
    latest_checkpoint_path,
    resolve_checkpoint_path,
    save_checkpoint,
)
from skull.train.trainer_pretrain import ErrorHandler, TrainingIntegrityError


def _build_fake_trainer(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = build_grad_scaler(enabled=False)
    return SimpleNamespace(
        cfg={},
        run_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=0,
        best_val_loss=1.5,
        run_name="test_run",
        device=torch.device("cpu"),
    )


def test_error_handler_restores_latest_checkpoint_when_model_turns_non_finite(
    tmp_path,
):
    trainer = _build_fake_trainer(tmp_path)
    trainer.step = 7
    saved_weight = trainer.model.weight.detach().clone()
    saved_bias = trainer.model.bias.detach().clone()
    save_checkpoint(
        tmp_path / "latest.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
        extra_state={"run_name": trainer.run_name},
    )

    with torch.no_grad():
        trainer.model.weight.fill_(float("nan"))

    handler = ErrorHandler(trainer)
    handler.handle_recoverable(TrainingIntegrityError("non-finite loss detected"), stage="train_step")

    assert trainer.step == 7
    assert torch.allclose(trainer.model.weight.detach(), saved_weight)
    assert torch.allclose(trainer.model.bias.detach(), saved_bias)

    errors = (tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(errors[-1])
    assert payload["action"] == "continue"
    assert payload["restored_from"] == str(tmp_path / "latest.pt")


def test_error_handler_refuses_to_save_non_finite_model(tmp_path):
    trainer = _build_fake_trainer(tmp_path)
    handler = ErrorHandler(trainer)

    with torch.no_grad():
        trainer.model.bias.fill_(float("nan"))

    with pytest.raises(
        TrainingIntegrityError,
        match="refusing to save non-finite model state",
    ):
        handler.ensure_model_is_savable()


def test_save_checkpoint_retries_transient_replace_permission_error(
    tmp_path,
    monkeypatch,
):
    trainer = _build_fake_trainer(tmp_path)
    path_type = type(tmp_path / "latest.pt")
    original_replace = path_type.replace
    replace_calls = 0

    def flaky_replace(self, target):
        nonlocal replace_calls
        if self.name.startswith(".latest.pt"):
            replace_calls += 1
            if replace_calls == 1:
                raise PermissionError("locked")
        return original_replace(self, target)

    monkeypatch.setattr(path_type, "replace", flaky_replace)

    save_checkpoint(
        tmp_path / "latest.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    assert replace_calls == 2
    assert latest_checkpoint_path(tmp_path) == tmp_path / "latest.pt"


def test_interrupt_checkpoint_is_discoverable_for_resume(tmp_path):
    trainer = _build_fake_trainer(tmp_path)
    trainer.step = 12
    handler = ErrorHandler(trainer)

    checkpoint_path = handler._save_error_checkpoint("interrupt")

    assert checkpoint_path == tmp_path / "interrupt_step_00000012.pt"
    assert not (tmp_path / "latest.pt").exists()
    assert latest_checkpoint_path(tmp_path) == checkpoint_path


def test_latest_checkpoint_path_falls_back_to_interrupt_checkpoint(tmp_path):
    trainer = _build_fake_trainer(tmp_path)
    trainer.step = 5
    save_checkpoint(
        tmp_path / "step_00000005.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    trainer.step = 9
    save_checkpoint(
        tmp_path / "interrupt_step_00000009.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    assert latest_checkpoint_path(tmp_path) == (
        tmp_path / "interrupt_step_00000009.pt"
    )


def test_latest_checkpoint_path_prefers_newer_interrupt_over_stale_latest(tmp_path):
    trainer = _build_fake_trainer(tmp_path)
    trainer.step = 77
    save_checkpoint(
        tmp_path / "latest.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    trainer.step = 120
    save_checkpoint(
        tmp_path / "interrupt_step_00000120.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    assert latest_checkpoint_path(tmp_path) == (
        tmp_path / "interrupt_step_00000120.pt"
    )


def test_latest_checkpoint_path_skips_corrupt_latest_when_interrupt_exists(tmp_path):
    (tmp_path / "latest.pt").write_bytes(b"not a checkpoint")

    trainer = _build_fake_trainer(tmp_path)
    trainer.step = 120
    save_checkpoint(
        tmp_path / "interrupt_step_00000120.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    assert latest_checkpoint_path(tmp_path) == (
        tmp_path / "interrupt_step_00000120.pt"
    )


def test_resolve_checkpoint_path_accepts_run_directory(tmp_path):
    trainer = _build_fake_trainer(tmp_path)
    trainer.step = 120
    save_checkpoint(
        tmp_path / "interrupt_step_00000120.pt",
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        step=trainer.step,
        best_val_loss=trainer.best_val_loss,
    )

    assert resolve_checkpoint_path(tmp_path) == (
        tmp_path / "interrupt_step_00000120.pt"
    )
