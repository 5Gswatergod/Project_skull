from __future__ import annotations

import argparse

from skull.cli.utils import (
    build_model_from_train_cfg,
    build_tokenizer_from_train_cfg,
    load_yaml,
    print_model_summary,
)
from skull.train import PretrainTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Project Skull pretraining CLI")
    parser.add_argument("--config", required=True, help="Path to pretrain yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    tokenizer = build_tokenizer_from_train_cfg(cfg)
    model = build_model_from_train_cfg(cfg)

    print_model_summary(model)

    trainer = PretrainTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
