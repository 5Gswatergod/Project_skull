import argparse
import sentencepiece as spm


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--tokenizer")

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    total = 0

    for line in open(args.input, encoding="utf-8"):

        ids = sp.encode(line)

        total += len(ids)

    print("tokens:", total)
    print("tokens (M):", total / 1e6)


if __name__ == "__main__":
    main()
