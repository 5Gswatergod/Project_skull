from pathlib import Path

import numpy as np
import pytest
import torch

from skull.data import MultiBinDataset


def _write_bin(path: Path, rows, dtype=np.uint32):
    arr = np.array(rows, dtype=dtype)
    arr.tofile(path)


def test_multi_bin_dataset_basic(tmp_path: Path):
    block_size = 4
    row_tokens = block_size + 1

    src1 = tmp_path / "src1.bin"
    src2 = tmp_path / "src2.bin"

    _write_bin(
        src1,
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
    )
    _write_bin(
        src2,
        [
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ],
    )

    ds = MultiBinDataset(
        sources=[
            {"name": "a", "paths": [str(src1)], "weight": 1.0},
            {"name": "b", "paths": [str(src2)], "weight": 1.0},
        ],
        block_size=block_size,
        dtype="uint32",
        row_tokens=row_tokens,
        nominal_size=100,
        seed=123,
    )

    assert len(ds) == 100

    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert "source_name" in item
    assert "source_id" in item
    assert "shard_id" in item

    assert item["input_ids"].shape[0] == block_size
    assert item["labels"].shape[0] == block_size
    assert torch.is_tensor(item["source_id"])
    assert torch.is_tensor(item["shard_id"])

    summary = ds.summary()
    assert summary["block_size"] == block_size
    assert len(summary["sources"]) == 2


def test_multi_bin_dataset_index_bounds(tmp_path: Path):
    block_size = 4
    row_tokens = block_size + 1

    src = tmp_path / "src.bin"
    _write_bin(src, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    ds = MultiBinDataset(
        sources=[{"name": "a", "paths": [str(src)], "weight": 1.0}],
        block_size=block_size,
        dtype="uint32",
        row_tokens=row_tokens,
        nominal_size=3,
        seed=123,
    )

    with pytest.raises(IndexError):
        _ = ds[-1]

    with pytest.raises(IndexError):
        _ = ds[3]
