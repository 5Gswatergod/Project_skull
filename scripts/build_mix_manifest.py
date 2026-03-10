import argparse
import json
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bins_dir")
    parser.add_argument("--output")

    args = parser.parse_args()

    sources = []

    for source in Path(args.bins_dir).iterdir():

        if not source.is_dir():
            continue

        bins = list(source.glob("train_*.bin"))

        if not bins:
            continue

        sources.append(
            {"name": source.name, "paths": [str(x) for x in bins], "weight": 1.0}
        )

    manifest = {"sources": sources}

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
