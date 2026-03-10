import sentencepiece as spm
from pathlib import Path


class Tokenizer:
    def __init__(self, model_path: Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

    def encode(self, text: str) -> list[int]:
        return self.sp.Encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.Decode(ids)

    @property
    def vocab_size(self) -> int:
        return int(self.sp.GetPieceSize())


def load_tokenizer(model_path: str | None = None) -> Tokenizer:
    path = (
        Path(model_path)
        if model_path
        else Path("data/tokenizer/zh_trad_en_100k_bpe.model")
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing SentencePiece model: {path.resolve()}")
    return Tokenizer(path)
