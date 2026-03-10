from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_script_module():
    script_path = Path("scripts/build_bins_multishard.py").resolve()
    spec = importlib.util.spec_from_file_location("build_bins_multishard", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummySP:
    def load(self, _path: str) -> None:
        return None

    def encode(self, text: str, out_type=int):  # noqa: ARG002
        n = len(text.strip())
        return [n] if n > 0 else []


def test_build_bins_multishard_meta_and_shard_count(tmp_path, monkeypatch):
    module = _load_script_module()

    input_path = tmp_path / "input.txt"
    input_path.write_text("a\nbb\nccc\n", encoding="utf-8")

    out_dir = tmp_path / "bins"
    tokenizer_path = tmp_path / "dummy.model"
    tokenizer_path.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(module.spm, "SentencePieceProcessor", lambda: _DummySP())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_bins_multishard.py",
            "--input",
            str(input_path),
            "--tokenizer",
            str(tokenizer_path),
            "--out_dir",
            str(out_dir),
            "--shard_tokens",
            "4",
        ],
    )

    module.main()

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["shards"] == 2
    assert meta["tokens"] == 6

    shard_files = sorted(out_dir.glob("train_*.bin"))
    assert len(shard_files) == 2
    assert sum(np.fromfile(p, dtype=np.uint32).size for p in shard_files) == 6
