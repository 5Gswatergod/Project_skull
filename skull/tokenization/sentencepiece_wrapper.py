from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing SentencePiece model: {self.model_path.resolve()}"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        return list(
            self.sp.encode(
                text,
                out_type=int,
                add_bos=bool(add_bos),
                add_eos=bool(add_eos),
            )
        )

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> str:
        token_ids = list(ids)
        if skip_special_tokens:
            specials = {sid for sid in [self.bos_id, self.eos_id, self.pad_id] if sid >= 0}
            token_ids = [tid for tid in token_ids if tid not in specials]
        return str(self.sp.decode(token_ids))

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    @property
    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    @property
    def pad_id(self) -> int:
        return int(self.sp.pad_id())

    @property
    def unk_id(self) -> int:
        return int(self.sp.unk_id())
