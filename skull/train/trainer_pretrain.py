from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from skull.data import MultiBinDataset, causal_lm_collate_fn
from skull.train.amp import build_grad_scaler
from skull.train.checkpointing import (
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from skull.train.accelerate_support import build_accelerator
from skull.train.losses import compute_causal_lm_loss, masked_token_accuracy
from skull.train.optimizer import build_optimizer
from skull.train.schedulers import build_lr_scheduler
from skull.train.stop import StopRequested


def _resolve_device(requested: str | None) -> torch.device:
    name = str(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available, falling back to CPU.")
        name = "cpu"
    return torch.device(name)


class TrainingIntegrityError(RuntimeError):
    """Raised when a train step would produce an unsafe model state."""


class ErrorHandler:
    def __init__(self, trainer: "PretrainTrainer") -> None:
        self.trainer = trainer
        self.errors_path = trainer.run_dir / "errors.jsonl"
        self.max_consecutive_errors = int(
            trainer.cfg.get("max_consecutive_errors", 3)
        )
        self.save_error_checkpoint = bool(
            trainer.cfg.get("save_error_checkpoint", True)
        )
        self.consecutive_errors = 0

    @staticmethod
    def is_recoverable(exc: BaseException) -> bool:
        if isinstance(exc, TrainingIntegrityError):
            return True
        if not isinstance(exc, RuntimeError):
            return False
        return "out of memory" in str(exc).lower()

    def note_success(self) -> None:
        self.consecutive_errors = 0

    def _model(self):
        getter = getattr(self.trainer, "unwrapped_model", None)
        if callable(getter):
            return getter()
        return self.trainer.model

    def _is_main_process(self) -> bool:
        return bool(getattr(self.trainer, "is_main_process", True))

    def ensure_finite_loss(self, loss: torch.Tensor) -> None:
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise TrainingIntegrityError("non-finite loss detected")

    def ensure_finite_grad_norm(self, grad_norm) -> None:
        if isinstance(grad_norm, torch.Tensor):
            is_finite = bool(torch.isfinite(grad_norm.detach()).all().item())
            value = float(grad_norm.detach().item())
        else:
            value = float(grad_norm)
            is_finite = math.isfinite(value)

        if not is_finite:
            raise TrainingIntegrityError(
                f"non-finite gradient norm detected: {value}"
            )

    def ensure_model_is_savable(self) -> None:
        bad_tensor = self._find_non_finite_model_tensor()
        if bad_tensor is not None:
            raise TrainingIntegrityError(
                f"refusing to save non-finite model state: {bad_tensor}"
            )

    def handle_recoverable(self, exc: BaseException, *, stage: str) -> None:
        restored_from = None
        bad_tensor = self._find_non_finite_model_tensor()

        if bad_tensor is not None:
            restored_from = self._restore_latest_checkpoint()
            if restored_from is None:
                self._reset_runtime_state()
                self._record_error(
                    stage=stage,
                    exc=exc,
                    action="abort",
                    bad_tensor=bad_tensor,
                    restored_from=None,
                    checkpoint_path=None,
                )
                raise RuntimeError(
                    f"Model state is invalid ({bad_tensor}) and no checkpoint is "
                    "available for recovery."
                ) from exc

        self._reset_runtime_state()
        self.consecutive_errors += 1
        self._record_error(
            stage=stage,
            exc=exc,
            action="continue",
            bad_tensor=bad_tensor,
            restored_from=restored_from,
            checkpoint_path=None,
        )

        if self.consecutive_errors >= self.max_consecutive_errors:
            raise RuntimeError(
                f"Exceeded max_consecutive_errors={self.max_consecutive_errors} "
                f"at step {self.trainer.step}."
            ) from exc

    def handle_fatal(
        self,
        exc: BaseException,
        *,
        stage: str,
        save_tag: str = "error",
    ) -> None:
        checkpoint_path = self._save_error_checkpoint(save_tag)
        self._reset_runtime_state()
        self._record_error(
            stage=stage,
            exc=exc,
            action="raise",
            bad_tensor=self._find_non_finite_model_tensor(),
            restored_from=None,
            checkpoint_path=checkpoint_path,
        )

    def _reset_runtime_state(self) -> None:
        self.trainer.optimizer.zero_grad(set_to_none=True)
        self.trainer.model.train()
        if self.trainer.device.type == "cuda":
            torch.cuda.empty_cache()

    def _find_non_finite_model_tensor(self) -> Optional[str]:
        model = self._model()
        for name, param in model.named_parameters():
            if not bool(torch.isfinite(param.detach()).all().item()):
                return f"parameter:{name}"

        for name, buffer in model.named_buffers():
            if not bool(torch.isfinite(buffer.detach()).all().item()):
                return f"buffer:{name}"

        return None

    def _restore_latest_checkpoint(self) -> Optional[Path]:
        ckpt_path = latest_checkpoint_path(self.trainer.run_dir)
        if ckpt_path is None:
            return None

        state = load_checkpoint(
            ckpt_path,
            model=self._model(),
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler,
            scaler=self.trainer.scaler,
            map_location="cpu",
        )
        self.trainer.step = int(state.get("step", 0))
        best_val = state.get("best_val_loss")
        if best_val is not None:
            self.trainer.best_val_loss = float(best_val)
        self.trainer.model.train()
        return ckpt_path

    def _save_error_checkpoint(self, save_tag: str) -> Optional[Path]:
        if not self.save_error_checkpoint:
            return None
        if not self._is_main_process():
            return None

        if self._find_non_finite_model_tensor() is not None:
            return None

        path = self.trainer.run_dir / f"{save_tag}_step_{self.trainer.step:08d}.pt"
        save_checkpoint(
            path,
            model=self._model(),
            optimizer=self.trainer.optimizer,
            scheduler=self.trainer.scheduler,
            scaler=self.trainer.scaler,
            step=self.trainer.step,
            best_val_loss=self.trainer.best_val_loss,
            extra_state={"run_name": self.trainer.run_name, "save_tag": save_tag},
        )
        # latest_checkpoint_path also scans interrupt checkpoints, so requested stops
        # do not need a second full write to latest.pt.
        return path

    def _record_error(
        self,
        *,
        stage: str,
        exc: BaseException,
        action: str,
        bad_tensor: Optional[str],
        restored_from: Optional[Path],
        checkpoint_path: Optional[Path],
    ) -> None:
        payload = {
            "step": self.trainer.step,
            "stage": stage,
            "action": action,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "consecutive_errors": self.consecutive_errors,
            "bad_tensor": bad_tensor,
            "restored_from": str(restored_from) if restored_from else None,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "time": time.time(),
        }
        if not self._is_main_process():
            return
        with open(self.errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[error] {payload}")


class PretrainTrainer:
    def __init__(self, cfg: dict, model, tokenizer=None) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = build_accelerator(cfg)
        self.accelerator_enabled = self.accelerator is not None

        self.run_name = cfg.get("run_name", "skull_pretrain")
        self.run_dir = Path(cfg.get("run_dir", f"runs/pretrain/{self.run_name}"))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = (
            self.accelerator.device
            if self.accelerator_enabled
            else _resolve_device(cfg.get("device"))
        )
        self.dtype_name = str(cfg.get("mixed_precision", "fp16")).lower()
        self.use_amp = (
            not self.accelerator_enabled
            and self.device.type == "cuda"
            and self.dtype_name in {"fp16", "bf16"}
        )

        self.batch_size = int(cfg.get("batch_size", 8))
        self.grad_accum = int(cfg.get("grad_accum", 1))
        self.max_steps = int(cfg.get("max_steps", 1000))
        self.log_every = int(cfg.get("log_every", 10))
        self.eval_every = int(cfg.get("eval_every", 200))
        self.save_every = int(cfg.get("save_every", self.eval_every))
        self.sample_every = int(cfg.get("sample_every", 0))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.seed = int(cfg.get("seed", 42))
        self.resume = bool(cfg.get("resume", True))
        self.train_nominal_size = int(cfg.get("train_nominal_size", 1_000_000))
        self.val_nominal_size = int(cfg.get("val_nominal_size", 50_000))
        self.eval_batches = int(cfg.get("eval_batches", 50))

        self.block_size = int(cfg["block_size"])
        self.bin_dtype = str(cfg.get("bin_dtype", "uint32"))
        self.row_tokens = int(cfg.get("row_tokens", self.block_size + 1))

        if not self.accelerator_enabled:
            self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_lr_scheduler(self.optimizer, cfg)

        if self.accelerator_enabled:
            self.scaler = getattr(self.accelerator, "scaler", None)
        else:
            scaler_enabled = self.device.type == "cuda" and self.dtype_name == "fp16"
            self.scaler = build_grad_scaler(enabled=scaler_enabled)

        self.step = 0
        self.best_val_loss = float("inf")

        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_val_loader()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.error_handler = ErrorHandler(self)
        self.stop_request_path = self._resolve_stop_request_path()

        if self.resume:
            self._try_resume()
        if self.accelerator_enabled:
            self._prepare_with_accelerate()

    @property
    def is_main_process(self) -> bool:
        return bool(
            self.accelerator is None or getattr(self.accelerator, "is_main_process", True)
        )

    def unwrapped_model(self):
        if self.accelerator is None:
            return self.model
        return self.accelerator.unwrap_model(self.model)

    def _prepare_with_accelerate(self) -> None:
        if self.val_loader is None:
            self.model, self.optimizer, self.train_loader, self.scheduler = (
                self.accelerator.prepare(
                    self.model,
                    self.optimizer,
                    self.train_loader,
                    self.scheduler,
                )
            )
            return

        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = (
            self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_loader,
                self.val_loader,
                self.scheduler,
            )
        )

    def _amp_dtype(self):
        if self.dtype_name == "bf16":
            return torch.bfloat16
        return torch.float16

    def _autocast_context(self):
        if self.accelerator_enabled:
            return self.accelerator.autocast()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype(),
            enabled=self.use_amp,
        )

    def _print(self, payload) -> None:
        if self.is_main_process:
            print(payload)

    def _resolve_stop_request_path(self) -> Optional[Path]:
        raw_path = os.environ.get("SKULL_STOP_REQUEST_PATH", "").strip()
        if not raw_path:
            return None
        return Path(raw_path)

    def _raise_if_stop_requested(self) -> None:
        if self.stop_request_path is None:
            return
        if self.stop_request_path.exists():
            raise StopRequested(self.stop_request_path)

    def _build_train_loader(self) -> DataLoader:
        ds = MultiBinDataset(
            sources=self.cfg["train_sources"],
            block_size=self.block_size,
            dtype=self.bin_dtype,
            row_tokens=self.row_tokens,
            nominal_size=self.train_nominal_size,
            seed=self.seed,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=causal_lm_collate_fn,
        )

    def _build_val_loader(self) -> Optional[DataLoader]:
        val_sources = self.cfg.get("val_sources")
        if not val_sources:
            return None

        ds = MultiBinDataset(
            sources=val_sources,
            block_size=self.block_size,
            dtype=self.bin_dtype,
            row_tokens=self.row_tokens,
            nominal_size=self.val_nominal_size,
            seed=self.seed + 777,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=causal_lm_collate_fn,
        )

    def _try_resume(self) -> None:
        ckpt_path = latest_checkpoint_path(self.run_dir)
        if ckpt_path is None:
            return

        state = load_checkpoint(
            ckpt_path,
            model=self.unwrapped_model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location="cpu",
        )
        self.step = int(state.get("step", 0))
        best_val = state.get("best_val_loss")
        if best_val is not None:
            self.best_val_loss = float(best_val)

        self._print(f"[resume] loaded checkpoint: {ckpt_path} step={self.step}")

    def _move_batch(self, batch: dict) -> dict:
        out = {}
        for key, val in batch.items():
            out[key] = (
                val.to(self.device, non_blocking=True) if torch.is_tensor(val) else val
            )
        return out

    def _forward_loss(self, batch: dict):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        with self._autocast_context():
            outputs = self.model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = compute_causal_lm_loss(logits, labels)
            acc = masked_token_accuracy(logits.detach(), labels)

        return loss, logits, acc

    def _count_batch_tokens(self, batch: dict) -> int:
        labels = batch.get("labels")
        if torch.is_tensor(labels):
            return int((labels != -100).sum().item())

        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids):
            return int(input_ids.numel())

        return 0

    @staticmethod
    def _grad_norm_value(grad_norm) -> float:
        if isinstance(grad_norm, torch.Tensor):
            return float(grad_norm.detach().item())
        return float(grad_norm)

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        count = 0

        for i, batch in enumerate(self.val_loader):
            if i >= self.eval_batches:
                break
            batch = self._move_batch(batch)
            loss, _, acc = self._forward_loss(batch)
            if self.accelerator_enabled:
                packed = torch.tensor(
                    [[float(loss.item()), float(acc), 1.0]],
                    device=self.device,
                )
                gathered = self.accelerator.gather_for_metrics(packed)
                total_loss += float(gathered[:, 0].sum().item())
                total_acc += float(gathered[:, 1].sum().item())
                count += int(gathered[:, 2].sum().item())
            else:
                total_loss += float(loss.item())
                total_acc += float(acc)
                count += 1

        if count == 0:
            return {}

        avg_loss = total_loss / count
        return {
            "val_loss": avg_loss,
            "val_ppl": math.exp(avg_loss) if avg_loss < 20 else float("inf"),
            "val_acc": total_acc / count,
        }

    @torch.no_grad()
    def sample(self) -> None:
        if not self.is_main_process:
            return
        model = self.unwrapped_model()
        if self.tokenizer is None or not hasattr(model, "generate"):
            return

        prompts = self.cfg.get("sample_prompts", ["Project Skull"])
        max_new_tokens = int(self.cfg.get("sample_max_new_tokens", 64))
        sample_dir = self.run_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        for i, prompt in enumerate(prompts):
            input_ids = self.tokenizer.encode(str(prompt), add_bos=False, add_eos=False)
            x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=float(self.cfg.get("sample_temperature", 1.0)),
                top_k=self.cfg.get("sample_top_k", None),
                eos_id=getattr(self.tokenizer, "eos_id", None),
            )

            text = self.tokenizer.decode(y[0].tolist(), skip_special_tokens=False)
            with open(
                sample_dir / f"step_{self.step:08d}_{i}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text)

    def _write_metrics(self, payload: dict) -> None:
        if not self.is_main_process:
            return
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save(self, is_best: bool = False) -> None:
        self.error_handler.ensure_model_is_savable()
        if not self.is_main_process:
            return

        model = self.unwrapped_model()
        latest_path = self.run_dir / "latest.pt"
        step_path = self.run_dir / f"step_{self.step:08d}.pt"

        save_checkpoint(
            step_path,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.step,
            best_val_loss=self.best_val_loss,
            extra_state={"run_name": self.run_name},
        )
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.step,
            best_val_loss=self.best_val_loss,
            extra_state={"run_name": self.run_name},
        )

        if is_best:
            save_checkpoint(
                self.run_dir / "best.pt",
                model=model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                step=self.step,
                best_val_loss=self.best_val_loss,
                extra_state={"run_name": self.run_name},
            )

    def train(self) -> None:
        self.model.train()
        loader_iter = iter(self.train_loader)

        running_loss = 0.0
        running_acc = 0.0
        running_grad_norm = 0.0
        running_micro_steps = 0
        running_update_steps = 0
        running_tokens = 0
        total_tokens = 0
        train_start = time.time()
        window_start = train_start

        while self.step < self.max_steps:
            try:
                self._raise_if_stop_requested()
            except KeyboardInterrupt as exc:
                self.error_handler.handle_fatal(
                    exc, stage="stop_requested", save_tag="interrupt"
                )
                raise

            self.optimizer.zero_grad(set_to_none=True)

            try:
                step_tokens = 0
                for micro_step in range(self.grad_accum):
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(self.train_loader)
                        batch = next(loader_iter)

                    batch = self._move_batch(batch)
                    step_tokens += self._count_batch_tokens(batch)
                    sync_context = nullcontext()
                    if self.accelerator_enabled and micro_step < (self.grad_accum - 1):
                        sync_context = self.accelerator.no_sync(self.model)
                    with sync_context:
                        loss, _, acc = self._forward_loss(batch)
                        self.error_handler.ensure_finite_loss(loss)
                        loss = loss / self.grad_accum
                        if self.accelerator_enabled:
                            self.accelerator.backward(loss)
                        else:
                            self.scaler.scale(loss).backward()

                    running_loss += float(loss.item()) * self.grad_accum
                    running_acc += float(acc)
                    running_micro_steps += 1

                max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
                if self.accelerator_enabled:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm
                    )
                else:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm
                    )
                self.error_handler.ensure_finite_grad_norm(grad_norm)
                grad_norm_value = self._grad_norm_value(grad_norm)

                if self.accelerator_enabled:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.scheduler.step()
                running_tokens += step_tokens
                total_tokens += step_tokens
            except KeyboardInterrupt as exc:
                self.error_handler.handle_fatal(
                    exc, stage="train_step", save_tag="interrupt"
                )
                raise
            except Exception as exc:
                if self.error_handler.is_recoverable(exc):
                    self.error_handler.handle_recoverable(exc, stage="train_step")
                    running_loss = 0.0
                    running_acc = 0.0
                    running_grad_norm = 0.0
                    running_micro_steps = 0
                    running_update_steps = 0
                    running_tokens = 0
                    window_start = time.time()
                    continue

                self.error_handler.handle_fatal(exc, stage="train_step")
                raise

            self.error_handler.note_success()
            self.step += 1
            running_grad_norm += grad_norm_value
            running_update_steps += 1

            try:
                self._raise_if_stop_requested()

                if self.step % self.log_every == 0:
                    now = time.time()
                    window_elapsed = max(now - window_start, 1e-9)
                    payload = {
                        "step": self.step,
                        "train_loss": running_loss / max(1, running_micro_steps),
                        "train_acc": running_acc / max(1, running_micro_steps),
                        "grad_norm": running_grad_norm / max(1, running_update_steps),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "elapsed_sec": now - train_start,
                        "window_elapsed_sec": window_elapsed,
                        "steps_per_sec": running_update_steps / window_elapsed,
                        "tokens_per_sec": running_tokens / window_elapsed,
                        "tokens_seen": total_tokens,
                    }
                    self._print(payload)
                    self._write_metrics(payload)
                    running_loss = 0.0
                    running_acc = 0.0
                    running_grad_norm = 0.0
                    running_micro_steps = 0
                    running_update_steps = 0
                    running_tokens = 0
                    window_start = now

                if self.val_loader is not None and self.step % self.eval_every == 0:
                    val_metrics = self.evaluate()
                    if val_metrics:
                        payload = {"step": self.step, **val_metrics}
                        self._print(payload)
                        self._write_metrics(payload)
                        is_best = val_metrics["val_loss"] < self.best_val_loss
                        if is_best:
                            self.best_val_loss = float(val_metrics["val_loss"])
                        self._save(is_best=is_best)
                    self.model.train()
                elif self.step % self.save_every == 0:
                    self._save(is_best=False)

                if self.sample_every > 0 and self.step % self.sample_every == 0:
                    self.sample()
                    self.model.train()
            except KeyboardInterrupt as exc:
                self.error_handler.handle_fatal(
                    exc, stage="post_step", save_tag="interrupt"
                )
                raise
            except Exception as exc:
                if self.error_handler.is_recoverable(exc):
                    self.error_handler.handle_recoverable(exc, stage="post_step")
                    running_loss = 0.0
                    running_acc = 0.0
                    running_grad_norm = 0.0
                    running_micro_steps = 0
                    running_update_steps = 0
                    running_tokens = 0
                    window_start = time.time()
                    continue

                self.error_handler.handle_fatal(exc, stage="post_step")
                raise

        try:
            self._save(is_best=False)
        except KeyboardInterrupt as exc:
            self.error_handler.handle_fatal(
                exc, stage="final_save", save_tag="interrupt"
            )
            raise
        except Exception as exc:
            if self.error_handler.is_recoverable(exc):
                self.error_handler.handle_recoverable(exc, stage="final_save")
                self._save(is_best=False)
                return

            self.error_handler.handle_fatal(exc, stage="final_save")
            raise
