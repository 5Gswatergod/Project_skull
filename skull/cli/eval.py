from __future__ import annotations

import argparse
import json
import math

import torch

from skull.cli.utils import (
    build_model_from_train_cfg,
    load_yaml,
)
from skull.eval.perplexity import evaluate_perplexity_from_cfg
from skull.train import load_checkpoint


def _resolve_device(requested: str | None) -> torch.device:
    name = str(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available, falling back to CPU.")
        name = "cpu"
    return torch.device(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Project Skull eval CLI")
    parser.add_argument("--config", required=True, help="Path to eval yaml")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument(
        "--print_json",
        action="store_true",
        help="Print full metrics as JSON",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    model = build_model_from_train_cfg(cfg)
    load_checkpoint(args.ckpt, model=model, map_location="cpu", strict=True)

    device = _resolve_device(cfg.get("device"))
    model.to(device)
    model.eval()

    metrics = evaluate_perplexity_from_cfg(model=model, cfg=cfg, device=device)
    summary = {
        "eval_loss": metrics["loss"],
        "eval_ppl": math.exp(metrics["loss"]) if metrics["loss"] < 20 else float("inf"),
        "eval_tokens": metrics["tokens"],
        "eval_batches": metrics["batches"],
    }

    if args.print_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(summary)


if __name__ == "__main__":
    main()
