import argparse
import numpy as np
import sentencepiece as spm
from pathlib import Path
import json


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--tokenizer")
    parser.add_argument("--out_dir")
    parser.add_argument("--shard_tokens", type=int, default=50_000_000)

    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    shard = 0
    buffer = []
    total = 0

    for line in open(args.input, encoding="utf-8"):

        ids = sp.encode(line)
        ids.append(2)

        buffer.extend(ids)

        if len(buffer) >= args.shard_tokens:

            arr = np.array(buffer, dtype=np.uint32)

            out = Path(args.out_dir) / f"train_{shard:03d}.bin"

            arr.tofile(out)

            total += len(arr)

            buffer = []

            shard += 1

            print("write shard", shard)

    if buffer:

        arr = np.array(buffer, dtype=np.uint32)

        out = Path(args.out_dir) / f"train_{shard:03d}.bin"

        arr.tofile(out)

        total += len(arr)

    meta = {"shards": shard + 1, "tokens": total}

    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
