from pathlib import Path

from .sentencepiece_wrapper import SentencePieceTokenizer


Tokenizer = SentencePieceTokenizer


def load_tokenizer(model_path: str | Path | None = None) -> SentencePieceTokenizer:
    path = (
        Path(model_path)
        if model_path
        else Path("data/tokenizer/zh_trad_en_100k_bpe.model")
    )
    return SentencePieceTokenizer(path)


def build_tokenizer(cfg: dict) -> SentencePieceTokenizer:
    model_path = cfg.get("tokenizer_model") or cfg.get("model_path")
    if not model_path:
        raise ValueError(
            "Tokenizer config must contain 'tokenizer_model' or 'model_path'"
        )
    return load_tokenizer(model_path)
