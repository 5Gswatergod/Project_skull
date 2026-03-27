import argparse
import json
import random
from pathlib import Path

import numpy as np
import sentencepiece as spm


def write_shard(out_dir, prefix, shard_idx, token_ids):
    arr = np.asarray(token_ids, dtype=np.uint32)
    out = out_dir / f"{prefix}_{shard_idx:03d}.bin"
    arr.tofile(out)
    return int(arr.size)


def flush(buffer, out_dir, prefix, shard_idx, shard_tokens):
    written_total = 0
    shards = 0

    while len(buffer) >= shard_tokens:
        chunk = buffer[:shard_tokens]
        written = write_shard(out_dir, prefix, shard_idx, chunk)
        print(f"write {prefix} shard {shard_idx:03d} tokens={written}")
        del buffer[:shard_tokens]
        shard_idx += 1
        written_total += written
        shards += 1

    return shard_idx, written_total, shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shard_tokens", type=int, default=50_000_000)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    train_buffer = []
    val_buffer = []

    train_idx = 0
    val_idx = 0

    train_tokens = 0
    val_tokens = 0

    train_shards = 0
    val_shards = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            ids = list(sp.encode(line, out_type=int))
            ids.append(2)

            if random.random() < args.val_ratio:
                val_buffer.extend(ids)
                val_idx, t, s = flush(
                    val_buffer, out_dir, "val", val_idx, args.shard_tokens
                )
                val_tokens += t
                val_shards += s
            else:
                train_buffer.extend(ids)
                train_idx, t, s = flush(
                    train_buffer, out_dir, "train", train_idx, args.shard_tokens
                )
                train_tokens += t
                train_shards += s

    # flush remaining
    if train_buffer:
        t = write_shard(out_dir, "train", train_idx, train_buffer)
        train_tokens += t
        train_shards += 1

    if val_buffer:
        t = write_shard(out_dir, "val", val_idx, val_buffer)
        val_tokens += t
        val_shards += 1

    meta = {
        "train_shards": train_shards,
        "val_shards": val_shards,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "dtype": "uint32",
        "eos_id": 2,
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
