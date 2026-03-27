import random
from pathlib import Path
import sentencepiece as spm

# =========================
# Config
# =========================
ROOT = Path(".")
TRAIN_TXT = ROOT / "data" / "clean" / "train_clean.txt"
OUT_DIR = ROOT / "data" / "tokenizer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PREFIX = OUT_DIR / "zh_trad_en_128k_bpe"
SAMPLE_TXT = OUT_DIR / "tokenizer_sample.txt"

VOCAB_SIZE = 100_000
MODEL_TYPE = "bpe"

MAX_SAMPLE_LINES = 2_000_000
INPUT_SENTENCE_SIZE = 2_000_000

MIN_LINE_CHARS = 5
MAX_LINE_CHARS = 4096

NUM_THREADS = 8
SEED = 42


def iter_clean_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) < MIN_LINE_CHARS:
                continue
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS]
            yield line


def build_sample_file(src: Path, dst: Path, max_lines: int) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing training text: {src.resolve()}")

    random.seed(SEED)
    sample = []
    seen = 0

    print(f"[sample] reading from: {src}")
    print(f"[sample] target lines: {max_lines:,}")

    for line in iter_clean_lines(src):
        seen += 1

        if len(sample) < max_lines:
            sample.append(line)
        else:
            j = random.randint(0, seen - 1)
            if j < max_lines:
                sample[j] = line

        if seen % 500_000 == 0:
            print(f"[sample] scanned {seen:,} lines, kept {len(sample):,}")

    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for line in sample:
            f.write(line + "\n")

    print(f"[sample] done")
    print(f"[sample] scanned total : {seen:,}")
    print(f"[sample] written lines : {len(sample):,}")
    print(f"[sample] saved to      : {dst}")


def train_tokenizer(input_txt: Path, model_prefix: Path) -> None:
    print("[spm] start training")
    print(f"[spm] input              : {input_txt}")
    print(f"[spm] model_prefix       : {model_prefix}")
    print(f"[spm] vocab_size         : {VOCAB_SIZE:,}")
    print(f"[spm] input_sentence_size: {INPUT_SENTENCE_SIZE:,}")
    print(f"[spm] max_sentence_length: {MAX_LINE_CHARS:,}")
    print(f"[spm] num_threads        : {NUM_THREADS}")

    kwargs = dict(
        input=str(input_txt),
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
        input_sentence_size=INPUT_SENTENCE_SIZE,
        shuffle_input_sentence=True,
        max_sentence_length=MAX_LINE_CHARS,
        split_digits=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=False,
        normalization_rule_name="nmt_nfkc",
        num_threads=NUM_THREADS,
    )

    try:
        spm.SentencePieceTrainer.Train(
            **kwargs,
            train_extremely_large_corpus=True,
        )
    except TypeError:
        print(
            "[spm] train_extremely_large_corpus not supported in this version, fallback."
        )
        spm.SentencePieceTrainer.Train(**kwargs)

    print("[spm] done")
    print("Saved:", str(model_prefix) + ".model")
    print("Saved:", str(model_prefix) + ".vocab")


def main():
    if not TRAIN_TXT.exists():
        raise FileNotFoundError(f"Missing training text: {TRAIN_TXT.resolve()}")

    build_sample_file(TRAIN_TXT, SAMPLE_TXT, MAX_SAMPLE_LINES)
    train_tokenizer(SAMPLE_TXT, MODEL_PREFIX)


if __name__ == "__main__":
    main()
