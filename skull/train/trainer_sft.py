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

from skull.data import PackedSFTDataset, sft_collate_fn
from skull.train.amp import build_grad_scaler
from skull.train.checkpointing import (
    latest_checkpoint_path,
    load_checkpoint,
    resolve_checkpoint_path,
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


class SFTTrainer:
    def __init__(self, cfg: dict, model, tokenizer) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = build_accelerator(cfg)
        self.accelerator_enabled = self.accelerator is not None

        self.run_name = cfg.get("run_name", "skull_sft")
        self.run_dir = Path(cfg.get("run_dir", f"runs/sft/{self.run_name}"))
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

        self.batch_size = int(cfg.get("batch_size", 4))
        self.grad_accum = int(cfg.get("grad_accum", 1))
        self.max_steps = int(cfg.get("max_steps", 1000))
        self.log_every = int(cfg.get("log_every", 10))
        self.eval_every = int(cfg.get("eval_every", 200))
        self.save_every = int(cfg.get("save_every", self.eval_every))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.resume = bool(cfg.get("resume", True))
        self.eval_batches = int(cfg.get("eval_batches", 50))
        self.max_seq_len = int(cfg.get("max_seq_len", 2048))

        if not self.accelerator_enabled:
            self.model.to(self.device)

        base_ckpt = cfg.get("base_ckpt")
        if base_ckpt:
            ckpt_path = resolve_checkpoint_path(base_ckpt)
            if ckpt_path is None:
                raise FileNotFoundError(f"base_ckpt does not exist: {base_ckpt}")
            load_checkpoint(
                ckpt_path,
                model=self.model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                map_location="cpu",
                strict=True,
            )
            self._print(f"[sft] loaded base checkpoint: {ckpt_path}")

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
        ds = PackedSFTDataset(
            jsonl_path=self.cfg["train_jsonl"],
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            assistant_only_loss=bool(self.cfg.get("assistant_only_loss", True)),
            packing=bool(self.cfg.get("packing", True)),
            add_bos=bool(self.cfg.get("add_bos", False)),
            add_eos=bool(self.cfg.get("add_eos", True)),
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=sft_collate_fn,
        )

    def _build_val_loader(self) -> Optional[DataLoader]:
        val_jsonl = self.cfg.get("val_jsonl")
        if not val_jsonl:
            return None

        ds = PackedSFTDataset(
            jsonl_path=val_jsonl,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            assistant_only_loss=bool(self.cfg.get("assistant_only_loss", True)),
            packing=False,
            add_bos=bool(self.cfg.get("add_bos", False)),
            add_eos=bool(self.cfg.get("add_eos", True)),
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=sft_collate_fn,
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
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
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
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        avg_acc = total_acc / count
        return {
            "val_loss": avg_loss,
            "val_ppl": ppl,
            "val_acc": avg_acc,
        }

    def _write_metrics(self, payload: dict) -> None:
        if not self.is_main_process:
            return
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save(self, is_best: bool = False) -> None:
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
            best_path = self.run_dir / "best.pt"
            save_checkpoint(
                best_path,
                model=model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                step=self.step,
                best_val_loss=self.best_val_loss,
                extra_state={"run_name": self.run_name},
            )

    def _save_interrupt(self) -> None:
        if not self.is_main_process:
            return

        model = self.unwrapped_model()
        save_checkpoint(
            self.run_dir / f"interrupt_step_{self.step:08d}.pt",
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.step,
            best_val_loss=self.best_val_loss,
            extra_state={"run_name": self.run_name, "save_tag": "interrupt"},
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

        try:
            while self.step < self.max_steps:
                self._raise_if_stop_requested()
                self.optimizer.zero_grad(set_to_none=True)

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
                grad_norm_value = self._grad_norm_value(grad_norm)

                if self.accelerator_enabled:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.scheduler.step()
                self.step += 1
                running_grad_norm += grad_norm_value
                running_update_steps += 1
                running_tokens += step_tokens
                total_tokens += step_tokens
                self._raise_if_stop_requested()

                if self.step % self.log_every == 0:
                    now = time.time()
                    window_elapsed = max(now - window_start, 1e-9)
                    avg_loss = running_loss / max(1, running_micro_steps)
                    avg_acc = running_acc / max(1, running_micro_steps)
                    lr = self.optimizer.param_groups[0]["lr"]

                    payload = {
                        "step": self.step,
                        "train_loss": avg_loss,
                        "train_acc": avg_acc,
                        "grad_norm": running_grad_norm / max(1, running_update_steps),
                        "lr": lr,
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

                        is_best = False
                        if val_metrics["val_loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["val_loss"]
                            is_best = True

                        self._save(is_best=is_best)
                    self.model.train()

                elif self.step % self.save_every == 0:
                    self._save(is_best=False)

            self._save(is_best=False)
        except StopRequested:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()
            self._save_interrupt()
            raise
        except KeyboardInterrupt:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()
            self._save(is_best=False)
            raise
