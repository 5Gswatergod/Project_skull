import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sentencepiece as spm

# =========================
# Config
# =========================
ROOT = Path(".")

OUT_DIR = ROOT / "data" / "tokenizer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 你可以把英文 / 中文 clean text 各自整理成這兩份
ENGLISH_TXT = ROOT / "data" / "clean" / "fineweb_clean.txt"
CHINESE_TXT = ROOT / "data" / "clean" / "novel_clean.txt"

# tokenizer sample 輸出
SAMPLE_TXT = OUT_DIR / "tokenizer_mix_zh70_en30.txt"
MODEL_PREFIX = OUT_DIR / "skull_zh_en_128k_bpe"

# vocab
VOCAB_SIZE = 100_000
MODEL_TYPE = "bpe"

# 抽樣總量
TOTAL_SAMPLE_LINES = 8_000_000
INPUT_SENTENCE_SIZE = 8_000_000

# 比例：中文 70%，英文 30% (如果你想要混合語料，調整這裡的比例)
SOURCE_RATIOS: Dict[str, float] = {
    "zh": 0.70,
    "en": 0.30,
}

SOURCE_PATHS: Dict[str, Path] = {
    "zh": CHINESE_TXT,
    "en": ENGLISH_TXT,
}

MIN_LINE_CHARS = 5
MAX_LINE_CHARS = 4096

NUM_THREADS = 10
SEED = 42


# =========================
# Utils
# =========================
def normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("SOURCE_RATIOS sum must be > 0")
    return {k: v / total for k, v in ratios.items()}


def compute_target_lines(
    total_lines: int,
    ratios: Dict[str, float],
) -> Dict[str, int]:
    ratios = normalize_ratios(ratios)
    items = list(ratios.items())

    targets: Dict[str, int] = {}
    acc = 0

    for i, (name, ratio) in enumerate(items):
        if i < len(items) - 1:
            n = int(total_lines * ratio)
            targets[name] = n
            acc += n
        else:
            targets[name] = total_lines - acc

    return targets


def iter_clean_lines(path: Path) -> Iterable[str]:
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


def reservoir_sample_lines(
    src: Path,
    max_lines: int,
    seed: int,
    source_name: str,
) -> List[str]:
    if not src.exists():
        raise FileNotFoundError(f"[{source_name}] missing input file: {src.resolve()}")

    rng = random.Random(seed)
    sample: List[str] = []
    seen = 0

    print(f"[sample:{source_name}] reading from : {src}")
    print(f"[sample:{source_name}] target lines : {max_lines:,}")

    for line in iter_clean_lines(src):
        seen += 1

        if len(sample) < max_lines:
            sample.append(line)
        else:
            j = rng.randint(0, seen - 1)
            if j < max_lines:
                sample[j] = line

        if seen % 500_000 == 0:
            print(
                f"[sample:{source_name}] scanned {seen:,} lines, kept {len(sample):,}"
            )

    print(f"[sample:{source_name}] done")
    print(f"[sample:{source_name}] scanned total : {seen:,}")
    print(f"[sample:{source_name}] written lines : {len(sample):,}")

    return sample


def write_mixed_sample(
    dst: Path,
    source_samples: Dict[str, List[str]],
    shuffle: bool = True,
    seed: int = SEED,
) -> None:
    mixed: List[Tuple[str, str]] = []

    for source_name, lines in source_samples.items():
        for line in lines:
            mixed.append((source_name, line))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(mixed)

    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for _, line in mixed:
            f.write(line + "\n")

    stats = {k: len(v) for k, v in source_samples.items()}
    total = sum(stats.values())

    print("[mix] done")
    print(f"[mix] total lines : {total:,}")
    for k, v in stats.items():
        pct = 100.0 * v / max(total, 1)
        print(f"[mix] {k:>6} : {v:,} ({pct:.2f}%)")
    print(f"[mix] saved to    : {dst}")


def build_balanced_sample_file(
    source_paths: Dict[str, Path],
    source_ratios: Dict[str, float],
    dst: Path,
    total_lines: int,
) -> None:
    targets = compute_target_lines(total_lines, source_ratios)

    print("[plan] sample plan")
    for name, n in targets.items():
        print(f"[plan] {name:>6} -> {n:,} lines")

    source_samples: Dict[str, List[str]] = {}

    for i, (name, path) in enumerate(source_paths.items()):
        if name not in targets:
            raise KeyError(f"Missing ratio for source: {name}")

        target_n = targets[name]
        source_seed = SEED + i * 1009
        lines = reservoir_sample_lines(
            src=path,
            max_lines=target_n,
            seed=source_seed,
            source_name=name,
        )
        source_samples[name] = lines

    write_mixed_sample(dst=dst, source_samples=source_samples, shuffle=True, seed=SEED)


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
    for name, path in SOURCE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"[{name}] missing input file: {path.resolve()}")

    build_balanced_sample_file(
        source_paths=SOURCE_PATHS,
        source_ratios=SOURCE_RATIOS,
        dst=SAMPLE_TXT,
        total_lines=TOTAL_SAMPLE_LINES,
    )
    train_tokenizer(SAMPLE_TXT, MODEL_PREFIX)


if __name__ == "__main__":
    main()
