from __future__ import annotations

from pathlib import Path
from typing import Optional

from .sentencepiece_wrapper import SentencePieceTokenizer


def load_tokenizer(model_path: str | Path) -> SentencePieceTokenizer:
    return SentencePieceTokenizer(model_path)


def build_tokenizer(cfg: dict) -> SentencePieceTokenizer:
    model_path = cfg.get("tokenizer_model") or cfg.get("model_path")
    if not model_path:
        raise ValueError(
            "Tokenizer config must contain 'tokenizer_model' or 'model_path'"
        )
    return load_tokenizer(model_path)
