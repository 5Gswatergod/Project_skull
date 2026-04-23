from __future__ import annotations

import argparse

from skull.cli.utils import (
    build_model_from_train_cfg,
    build_tokenizer_from_train_cfg,
    load_yaml,
    print_model_summary,
)
from skull.train.accelerate_support import is_primary_process_from_env
from skull.train import PretrainTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Project Skull pretraining CLI")
    parser.add_argument("--config", required=True, help="Path to pretrain yaml")
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Enable Hugging Face Accelerate support for this training run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.accelerate:
        cfg["use_accelerate"] = True

    tokenizer = build_tokenizer_from_train_cfg(cfg)
    model = build_model_from_train_cfg(cfg)
    if not cfg.get("use_accelerate") or is_primary_process_from_env():
        print_model_summary(model)

    trainer = PretrainTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
