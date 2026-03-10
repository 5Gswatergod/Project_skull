import argparse
import numpy as np
import sentencepiece as spm


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--tokenizer")
    parser.add_argument("--output")

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    tokens = []

    for line in open(args.input, encoding="utf-8"):

        ids = sp.encode(line)
        tokens.extend(ids + [2])

    arr = np.array(tokens, dtype=np.uint32)

    arr.tofile(args.output)

    print("tokens:", len(arr))


if __name__ == "__main__":
    main()
