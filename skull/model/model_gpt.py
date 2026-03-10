from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import CausalSelfAttention
from .config import GPTConfig
from .mlp import build_mlp
from .norms import build_norm


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = build_norm(
            config.norm,
            config.n_embd,
            bias=config.bias,
            eps=config.norm_eps,
        )
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(
            config.norm,
            config.n_embd,
            bias=config.bias,
            eps=config.norm_eps,
        )
        self.mlp = build_mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = (
            nn.Embedding(config.block_size, config.n_embd)
            if str(config.pos_encoding).lower() == "absolute"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = build_norm(
            config.norm,
            config.n_embd,
            bias=config.bias,
            eps=config.norm_eps,
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}"
            )

        bsz, seqlen = input_ids.shape
        if seqlen > self.config.block_size:
            raise ValueError(
                f"Sequence length {seqlen} exceeds block_size {self.config.block_size}"
            )

        x = self.wte(input_ids)
        if self.wpe is not None:
            pos = torch.arange(seqlen, device=input_ids.device)
            x = x + self.wpe(pos)[None, :, :]
        x = self.drop(x)

        for block in self.h:
            x = self._forward_block(block, x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} does not match "
                    f"input_ids {tuple(input_ids.shape)}"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}"
            )

        x = input_ids
        temp = max(float(temperature), 1e-6)
        self.eval()

        for _ in range(int(max_new_tokens)):
            x_cond = x[:, -self.config.block_size :]
            out = self(x_cond)
            logits = out["logits"][:, -1, :] / temp

            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), logits.size(-1))
                vals, _ = torch.topk(logits, k=k, dim=-1)
                kth = vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < kth,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

            if eos_id is not None and torch.all(next_token.squeeze(-1) == int(eos_id)):
                break

        return x
