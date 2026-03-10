from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class BinSliceInfo:
    path: str
    num_rows: int
    row_tokens: int
    dtype: str


class BlockBinDataset(Dataset):
    """
    讀取 row-packed bin：
    每 row 長度 = block_size + 1
    x = row[:-1]
    y = row[1:]
    """

    def __init__(
        self,
        path: str | Path,
        block_size: int,
        dtype: str = "uint32",
        row_tokens: Optional[int] = None,
        drop_last_incomplete: bool = True,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.block_size = int(block_size)
        self.row_tokens = int(row_tokens or (block_size + 1))
        self.dtype = np.dtype(dtype)
        self.drop_last_incomplete = drop_last_incomplete

        if self.row_tokens != self.block_size + 1:
            raise ValueError(
                f"Expected row_tokens == block_size + 1, got "
                f"{self.row_tokens} vs {self.block_size + 1}"
            )

        file_path = Path(self.path)
        if not file_path.exists():
            raise FileNotFoundError(f"Bin file not found: {self.path}")

        self._data = np.memmap(file_path, dtype=self.dtype, mode="r")
        total_tokens = int(self._data.shape[0])

        if total_tokens < self.row_tokens:
            self.num_rows = 0
        else:
            if drop_last_incomplete:
                self.num_rows = total_tokens // self.row_tokens
            else:
                if total_tokens % self.row_tokens != 0:
                    raise ValueError(
                        f"Bin size not divisible by row_tokens: "
                        f"{total_tokens} % {self.row_tokens} != 0"
                    )
                self.num_rows = total_tokens // self.row_tokens

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_rows:
            raise IndexError(idx)

        start = idx * self.row_tokens
        end = start + self.row_tokens
        row = np.asarray(self._data[start:end], dtype=np.int64)

        x = torch.from_numpy(row[:-1].copy()).long()
        y = torch.from_numpy(row[1:].copy()).long()

        return {
            "input_ids": x,
            "labels": y,
        }

    @property
    def info(self) -> BinSliceInfo:
        return BinSliceInfo(
            path=self.path,
            num_rows=self.num_rows,
            row_tokens=self.row_tokens,
            dtype=str(self.dtype),
        )
