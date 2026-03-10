from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from skull.model import GPT, GPTConfig
from skull.tokenization import load_tokenizer


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def maybe_load_model_config(train_cfg: dict) -> dict:
    """
    支援兩種寫法：
    1. train yaml 內直接有 model 欄位
    2. train yaml 指向 model_config 路徑
    """
    if "model" in train_cfg and isinstance(train_cfg["model"], dict):
        return train_cfg["model"]

    model_cfg_path = train_cfg.get("model_config")
    if model_cfg_path:
        return load_yaml(model_cfg_path)

    raise ValueError("Train config must contain either 'model' or 'model_config'")


def build_model_from_train_cfg(train_cfg: dict) -> GPT:
    model_cfg = maybe_load_model_config(train_cfg)
    config = GPTConfig.from_dict(model_cfg)
    return GPT(config)


def build_tokenizer_from_train_cfg(train_cfg: dict):
    tok_path = train_cfg.get("tokenizer_model")
    if not tok_path:
        raise ValueError("Train config missing 'tokenizer_model'")
    return load_tokenizer(tok_path)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module) -> None:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    print(
        {
            "model": model.__class__.__name__,
            "total_params": total,
            "trainable_params": trainable,
        }
    )
