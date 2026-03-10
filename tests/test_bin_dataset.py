from pathlib import Path

import numpy as np
import torch

from skull.data import BlockBinDataset


def test_block_bin_dataset_basic(tmp_path: Path):
    block_size = 4
    row_tokens = block_size + 1

    rows = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            10,
            11,
            12,
            13,
            14,
        ],
        dtype=np.uint32,
    )
    path = tmp_path / "toy.bin"
    rows.tofile(path)

    ds = BlockBinDataset(
        path=path,
        block_size=block_size,
        dtype="uint32",
        row_tokens=row_tokens,
    )

    assert len(ds) == 2

    item0 = ds[0]
    assert set(item0.keys()) == {"input_ids", "labels"}
    assert torch.equal(item0["input_ids"], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(item0["labels"], torch.tensor([2, 3, 4, 5]))

    item1 = ds[1]
    assert torch.equal(item1["input_ids"], torch.tensor([10, 11, 12, 13]))
    assert torch.equal(item1["labels"], torch.tensor([11, 12, 13, 14]))
