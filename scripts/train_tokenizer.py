import argparse
import sentencepiece as spm


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--model_prefix")
    parser.add_argument("--vocab_size", type=int, default=100000)

    args = parser.parse_args()

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )


if __name__ == "__main__":
    main()
