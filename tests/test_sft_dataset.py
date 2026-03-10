from __future__ import annotations

import json
from pathlib import Path

import torch

from skull.data import PackedSFTDataset
from skull.data.packed_sft_dataset import IGNORE_INDEX


class DummyTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 127 + 3 for ch in text]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prompt_response_assistant_only_loss_masking(tmp_path: Path):
    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [{"prompt": "hello", "response": "world"}])
    tok = DummyTokenizer()

    ds = PackedSFTDataset(
        jsonl_path=data_path,
        tokenizer=tok,
        max_seq_len=128,
        assistant_only_loss=True,
        packing=False,
        add_bos=False,
        add_eos=True,
    )
    item = ds[0]

    prompt_ids = tok.encode(ds.user_tag + "hello\n")
    resp_ids = tok.encode(ds.assistant_tag + "world")

    labels = item["labels"]
    assert torch.all(labels[: len(prompt_ids)] == IGNORE_INDEX)
    assert torch.equal(
        labels[len(prompt_ids) : len(prompt_ids) + len(resp_ids)],
        torch.tensor(resp_ids, dtype=torch.long),
    )


def test_prompt_response_full_loss_includes_prompt_tokens(tmp_path: Path):
    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(data_path, [{"prompt": "hello", "response": "world"}])
    tok = DummyTokenizer()

    ds = PackedSFTDataset(
        jsonl_path=data_path,
        tokenizer=tok,
        max_seq_len=128,
        assistant_only_loss=False,
        packing=False,
        add_bos=False,
        add_eos=True,
    )
    item = ds[0]

    prompt_ids = tok.encode(ds.user_tag + "hello\n")
    labels = item["labels"]
    assert torch.equal(
        labels[: len(prompt_ids)],
        torch.tensor(prompt_ids, dtype=torch.long),
    )
