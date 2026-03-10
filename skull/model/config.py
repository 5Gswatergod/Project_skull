from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    dropout: float = 0.0
    bias: bool = False

    norm: str = "layernorm"  # layernorm | rmsnorm
    norm_eps: float = 1e-5

    pos_encoding: str = "absolute"  # absolute | rope
    rope_base: float = 10000.0

    mlp_type: str = "gelu"  # gelu | swiglu
    mlp_hidden_mult: float = 4.0

    tie_word_embeddings: bool = True
    use_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd must be divisible by n_head, got {self.n_embd} / {self.n_head}"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_dict(cls, cfg: dict) -> "GPTConfig":
        return cls(**cfg)

    def to_dict(self) -> dict:
        return self.__dict__.copy()
