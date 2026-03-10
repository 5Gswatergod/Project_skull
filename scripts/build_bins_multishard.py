import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


def _write_shard(out_dir: Path, shard_idx: int, token_ids: list[int]) -> int:
    arr = np.asarray(token_ids, dtype=np.uint32)
    out = out_dir / f"train_{shard_idx:03d}.bin"
    arr.tofile(out)
    return int(arr.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shard_tokens", type=int, default=50_000_000)
    args = parser.parse_args()

    if args.shard_tokens <= 0:
        raise ValueError(f"--shard_tokens must be > 0, got {args.shard_tokens}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    buffer: list[int] = []
    shard_idx = 0
    written_shards = 0
    total_tokens = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            ids = list(sp.encode(line, out_type=int))
            ids.append(2)
            buffer.extend(ids)

            while len(buffer) >= args.shard_tokens:
                chunk = buffer[: args.shard_tokens]
                written = _write_shard(out_dir, shard_idx, chunk)
                total_tokens += written
                written_shards += 1
                print(f"write shard {shard_idx:03d} tokens={written}")
                shard_idx += 1
                del buffer[: args.shard_tokens]

    if buffer:
        written = _write_shard(out_dir, shard_idx, buffer)
        total_tokens += written
        written_shards += 1
        print(f"write shard {shard_idx:03d} tokens={written}")

    meta = {
        "shards": written_shards,
        "tokens": total_tokens,
        "shard_tokens": int(args.shard_tokens),
        "dtype": "uint32",
        "eos_id": 2,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
