import argparse
from pathlib import Path
import json


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", default=None)
    args = parser.parse_args()

    total = 0

    with open(args.output, "w", encoding="utf-8") as out:

        for path in args.inputs:

            for line in read_lines(path):
                out.write(line + "\n")
                total += 1

    if args.meta:
        meta = {"lines": total, "sources": args.inputs}
        with open(args.meta, "w") as f:
            json.dump(meta, f, indent=2)

    print("done lines:", total)


if __name__ == "__main__":
    main()
