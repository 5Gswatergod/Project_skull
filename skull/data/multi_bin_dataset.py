from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .block_bin_dataset import BlockBinDataset


@dataclass
class SourceConfig:
    name: str
    paths: list[str]
    weight: float = 1.0


class MultiBinDataset(Dataset):
    """
    多 source、多 shard 的 map-style dataset。

    設計理念：
    - __len__ 用 nominal_size 控制，方便 trainer 以 step 為中心訓練
    - __getitem__(idx) 不直接對應固定 row，而是使用 idx + seed 產生穩定抽樣
    - 先抽 source，再抽 shard，再抽 row
    """

    def __init__(
        self,
        sources: Sequence[dict | SourceConfig],
        block_size: int,
        dtype: str = "uint32",
        row_tokens: Optional[int] = None,
        nominal_size: int = 1_000_000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not sources:
            raise ValueError("sources must not be empty")

        self.block_size = int(block_size)
        self.row_tokens = int(row_tokens or (block_size + 1))
        self.dtype = dtype
        self.nominal_size = int(nominal_size)
        self.seed = int(seed)

        if self.nominal_size <= 0:
            raise ValueError(f"nominal_size must be > 0, got {self.nominal_size}")

        self.sources: list[dict] = []
        self.source_names: list[str] = []
        self.source_weights: np.ndarray
        self.shard_datasets: list[list[BlockBinDataset]] = []
        self.shard_lengths: list[np.ndarray] = []

        src_weights = []

        for src in sources:
            if isinstance(src, SourceConfig):
                src = {
                    "name": src.name,
                    "paths": src.paths,
                    "weight": src.weight,
                }

            name = src["name"]
            paths = [str(Path(p)) for p in src["paths"]]
            weight = float(src.get("weight", 1.0))

            if weight <= 0:
                raise ValueError(f"Source weight must be > 0, got {weight} for {name}")

            datasets = [
                BlockBinDataset(
                    path=p,
                    block_size=self.block_size,
                    dtype=self.dtype,
                    row_tokens=self.row_tokens,
                )
                for p in paths
            ]
            datasets = [ds for ds in datasets if len(ds) > 0]
            if not datasets:
                raise ValueError(f"Source '{name}' has no non-empty shards")

            lengths = np.array([len(ds) for ds in datasets], dtype=np.float64)
            lengths = lengths / lengths.sum()

            self.sources.append(
                {
                    "name": name,
                    "paths": paths,
                    "weight": weight,
                }
            )
            self.source_names.append(name)
            src_weights.append(weight)
            self.shard_datasets.append(datasets)
            self.shard_lengths.append(lengths)

        self.source_weights = np.array(src_weights, dtype=np.float64)
        self.source_weights = self.source_weights / self.source_weights.sum()

    def __len__(self) -> int:
        return self.nominal_size

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        # 讓每個 idx 對應穩定的抽樣結果
        return np.random.default_rng(self.seed + int(idx))

    def _sample_source_index(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.sources), p=self.source_weights))

    def _sample_shard_index(self, source_idx: int, rng: np.random.Generator) -> int:
        p = self.shard_lengths[source_idx]
        return int(rng.choice(len(p), p=p))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.nominal_size:
            raise IndexError(idx)

        rng = self._rng_for_index(idx)

        src_idx = self._sample_source_index(rng)
        shard_idx = self._sample_shard_index(src_idx, rng)
        ds = self.shard_datasets[src_idx][shard_idx]

        row_idx = int(rng.integers(0, len(ds)))
        item = ds[row_idx]

        item["source_name"] = self.source_names[src_idx]
        item["source_id"] = torch.tensor(src_idx, dtype=torch.long)
        item["shard_id"] = torch.tensor(shard_idx, dtype=torch.long)
        return item

    def summary(self) -> dict:
        sources = []
        total_rows = 0
        for src, shard_sets in zip(self.sources, self.shard_datasets):
            rows = sum(len(ds) for ds in shard_sets)
            total_rows += rows
            sources.append(
                {
                    "name": src["name"],
                    "weight": src["weight"],
                    "num_shards": len(shard_sets),
                    "rows": rows,
                }
            )

        return {
            "block_size": self.block_size,
            "row_tokens": self.row_tokens,
            "nominal_size": self.nominal_size,
            "total_rows": total_rows,
            "sources": sources,
        }
