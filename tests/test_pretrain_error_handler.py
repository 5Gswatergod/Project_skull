from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from skull.train.amp import build_grad_scaler
from skull.train.checkpointing import save_checkpoint
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
