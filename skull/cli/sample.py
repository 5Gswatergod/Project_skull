from __future__ import annotations

import argparse

import torch

from skull.cli.utils import (
    build_model_from_train_cfg,
    build_tokenizer_from_train_cfg,
    load_yaml,
)
from skull.train import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Project Skull sample CLI")
    parser.add_argument("--config", required=True, help="Path to train/eval yaml")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    tokenizer = build_tokenizer_from_train_cfg(cfg)
    model = build_model_from_train_cfg(cfg)

    load_checkpoint(args.ckpt, model=model, map_location="cpu", strict=True)

    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    y = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=getattr(tokenizer, "eos_id", None),
    )

    text = tokenizer.decode(y[0].tolist(), skip_special_tokens=False)
    print(text)


if __name__ == "__main__":
    main()
