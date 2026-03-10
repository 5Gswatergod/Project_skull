from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


@dataclass
class SFTSample:
    input_ids: list[int]
    labels: list[int]


class PackedSFTDataset(Dataset):
    """
    支援兩種資料格式：
    1. {"messages":[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
    2. {"prompt":"...", "response":"..."}

    tokenizer 需提供：
    - encode(str) -> list[int]
    - bos_id / eos_id / pad_id（可選）
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_seq_len: int = 2048,
        assistant_only_loss: bool = True,
        packing: bool = True,
        add_bos: bool = False,
        add_eos: bool = True,
        user_tag: str = "<|user|>\n",
        assistant_tag: str = "<|assistant|>\n",
    ) -> None:
        super().__init__()
        self.jsonl_path = str(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.assistant_only_loss = bool(assistant_only_loss)
        self.packing = bool(packing)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag

        self.pad_id = int(getattr(tokenizer, "pad_id", 0))
        self.bos_id = getattr(tokenizer, "bos_id", None)
        self.eos_id = getattr(tokenizer, "eos_id", None)

        self.samples: list[SFTSample] = []
        self._load()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        input_ids = torch.tensor(sample.input_ids, dtype=torch.long)
        labels = torch.tensor(sample.labels, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def _load(self) -> None:
        raw_samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                raw_samples.append(self._encode_record(obj))

        if self.packing:
            self.samples = self._pack_samples(raw_samples)
        else:
            self.samples = [self._pad_or_truncate(s) for s in raw_samples]

    def _encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def _encode_record(self, obj: dict) -> SFTSample:
        if "messages" in obj:
            return self._encode_messages(obj["messages"])
        if "prompt" in obj and "response" in obj:
            return self._encode_prompt_response(obj["prompt"], obj["response"])
        raise ValueError("Unsupported SFT record format")

    def _encode_prompt_response(self, prompt: str, response: str) -> SFTSample:
        parts_ids: list[int] = []
        parts_labels: list[int] = []

        if self.add_bos and self.bos_id is not None:
            parts_ids.append(int(self.bos_id))
            parts_labels.append(IGNORE_INDEX)

        prompt_text = self.user_tag + prompt.strip() + "\n"
        resp_text = self.assistant_tag + response.strip()

        prompt_ids = self._encode_text(prompt_text)
        resp_ids = self._encode_text(resp_text)

        parts_ids.extend(prompt_ids)
        if self.assistant_only_loss:
            parts_labels.extend([IGNORE_INDEX] * len(prompt_ids))
        else:
            parts_labels.extend(prompt_ids)

        parts_ids.extend(resp_ids)
        parts_labels.extend(resp_ids)

        if self.add_eos and self.eos_id is not None:
            parts_ids.append(int(self.eos_id))
            parts_labels.append(int(self.eos_id))

        return SFTSample(parts_ids, parts_labels)

    def _encode_messages(self, messages: list[dict]) -> SFTSample:
        ids: list[int] = []
        labels: list[int] = []

        if self.add_bos and self.bos_id is not None:
            ids.append(int(self.bos_id))
            labels.append(IGNORE_INDEX)

        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()

            if role == "user":
                text = self.user_tag + content + "\n"
                token_ids = self._encode_text(text)
                ids.extend(token_ids)
                if self.assistant_only_loss:
                    labels.extend([IGNORE_INDEX] * len(token_ids))
                else:
                    labels.extend(token_ids)

            elif role == "assistant":
                text = self.assistant_tag + content + "\n"
                token_ids = self._encode_text(text)
                ids.extend(token_ids)
                labels.extend(token_ids)

            else:
                # system / tool / other roles：預設不計 loss
                text = f"<|{role}|>\n{content}\n"
                token_ids = self._encode_text(text)
                ids.extend(token_ids)
                if self.assistant_only_loss:
                    labels.extend([IGNORE_INDEX] * len(token_ids))
                else:
                    labels.extend(token_ids)

        if self.add_eos and self.eos_id is not None:
            ids.append(int(self.eos_id))
            labels.append(int(self.eos_id))

        return SFTSample(ids, labels)

    def _pad_or_truncate(self, sample: SFTSample) -> SFTSample:
        input_ids = sample.input_ids[: self.max_seq_len]
        labels = sample.labels[: self.max_seq_len]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        return SFTSample(input_ids, labels)

    def _pack_samples(self, samples: Iterable[SFTSample]) -> list[SFTSample]:
        packed: list[SFTSample] = []
        cur_ids: list[int] = []
        cur_labels: list[int] = []

        for sample in samples:
            sample_ids = sample.input_ids
            sample_labels = sample.labels

            if len(sample_ids) > self.max_seq_len:
                sample = self._pad_or_truncate(sample)
                packed.append(sample)
                continue

            if len(cur_ids) + len(sample_ids) > self.max_seq_len:
                packed.append(self._pad_or_truncate(SFTSample(cur_ids, cur_labels)))
                cur_ids = []
                cur_labels = []

            cur_ids.extend(sample_ids)
            cur_labels.extend(sample_labels)

        if cur_ids:
            packed.append(self._pad_or_truncate(SFTSample(cur_ids, cur_labels)))

        return packed
