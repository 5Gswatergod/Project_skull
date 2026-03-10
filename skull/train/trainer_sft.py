from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from skull.data import PackedSFTDataset, sft_collate_fn
from skull.train.checkpointing import (
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from skull.train.losses import compute_causal_lm_loss, masked_token_accuracy
from skull.train.optimizer import build_optimizer
from skull.train.schedulers import build_lr_scheduler


class SFTTrainer:
    def __init__(self, cfg: dict, model, tokenizer) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

        self.run_name = cfg.get("run_name", "skull_sft")
        self.run_dir = Path(cfg.get("run_dir", f"runs/sft/{self.run_name}"))
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype_name = str(cfg.get("mixed_precision", "fp16")).lower()
        self.use_amp = self.device.type == "cuda" and self.dtype_name in {
            "fp16",
            "bf16",
        }

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

        self.model.to(self.device)

        base_ckpt = cfg.get("base_ckpt")
        if base_ckpt:
            load_checkpoint(
                base_ckpt,
                model=self.model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                map_location="cpu",
                strict=True,
            )
            print(f"[sft] loaded base checkpoint: {base_ckpt}")

        self.optimizer = build_optimizer(self.model, cfg)
        self.scheduler = build_lr_scheduler(self.optimizer, cfg)

        scaler_enabled = self.device.type == "cuda" and self.dtype_name == "fp16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        self.step = 0
        self.best_val_loss = float("inf")

        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_val_loader()
        self.metrics_path = self.run_dir / "metrics.jsonl"

        if self.resume:
            self._try_resume()

    def _amp_dtype(self):
        if self.dtype_name == "bf16":
            return torch.bfloat16
        return torch.float16

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
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location="cpu",
        )
        self.step = int(state.get("step", 0))
        best_val = state.get("best_val_loss")
        if best_val is not None:
            self.best_val_loss = float(best_val)

        print(f"[resume] loaded checkpoint: {ckpt_path} step={self.step}")

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

        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype(),
            enabled=self.use_amp,
        ):
            outputs = self.model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = compute_causal_lm_loss(logits, labels)
            acc = masked_token_accuracy(logits.detach(), labels)

        return loss, logits, acc

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
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save(self, is_best: bool = False) -> None:
        latest_path = self.run_dir / "latest.pt"
        step_path = self.run_dir / f"step_{self.step:08d}.pt"

        save_checkpoint(
            latest_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.step,
            best_val_loss=self.best_val_loss,
            extra_state={"run_name": self.run_name},
        )

        save_checkpoint(
            step_path,
            model=self.model,
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
                model=self.model,
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
        running_micro_steps = 0
        wall_start = time.time()

        while self.step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(self.grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)

                batch = self._move_batch(batch)

                loss, _, acc = self._forward_loss(batch)
                loss = loss / self.grad_accum
                self.scaler.scale(loss).backward()

                running_loss += float(loss.item()) * self.grad_accum
                running_acc += float(acc)
                running_micro_steps += 1

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.step += 1

            if self.step % self.log_every == 0:
                elapsed = time.time() - wall_start
                avg_loss = running_loss / max(1, running_micro_steps)
                avg_acc = running_acc / max(1, running_micro_steps)
                lr = self.optimizer.param_groups[0]["lr"]

                payload = {
                    "step": self.step,
                    "train_loss": avg_loss,
                    "train_acc": avg_acc,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                }
                print(payload)
                self._write_metrics(payload)

                running_loss = 0.0
                running_acc = 0.0
                running_micro_steps = 0

            if self.val_loader is not None and self.step % self.eval_every == 0:
                val_metrics = self.evaluate()
                if val_metrics:
                    payload = {"step": self.step, **val_metrics}
                    print(payload)
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
