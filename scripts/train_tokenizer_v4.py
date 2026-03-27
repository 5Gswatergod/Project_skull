#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import sentencepiece as spm

# -------------------------
# Defaults for Skull-128k
# -------------------------
DEFAULT_OUT_DIR = Path("data") / "tokenizer"
DEFAULT_SAMPLE_NAME = "tokenizer_mix_128k.txt"
DEFAULT_MODEL_PREFIX = "skull_zh_en_128k_bpe"
DEFAULT_TOTAL_SAMPLE_LINES = 12_000_000
DEFAULT_INPUT_SENTENCE_SIZE = 10_000_000
DEFAULT_VOCAB_SIZE = 128_000
DEFAULT_MODEL_TYPE = "bpe"
DEFAULT_CHARACTER_COVERAGE = 0.9999
DEFAULT_MIN_LINE_CHARS = 5
DEFAULT_MAX_LINE_CHARS = 4096
DEFAULT_NUM_THREADS = 8
DEFAULT_SEED = 42
DEFAULT_PROGRESS_EVERY = 500_000
DEFAULT_CONTROL_TOKENS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|observation|>",
    "<|end|>",
    "<|text|>",
    "<|code|>",
    "<|quote|>",
    "<|tab|>",
    "<|newline|>",
]

_DIGIT_RE = re.compile(r"\d")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{11,}")
_URL_RE = re.compile(r"https?://|www\.")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    ratio: float
    weight: float = 1.0


@dataclass
class SampleStats:
    name: str
    target: int
    kept: int = 0
    scanned: int = 0
    accepted_clean: int = 0
    dropped_empty: int = 0
    dropped_short: int = 0
    dropped_low_alpha: int = 0
    dropped_repetitive: int = 0
    dropped_duplicate: int = 0
    dropped_weight: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "target": self.target,
            "kept": self.kept,
            "scanned": self.scanned,
            "accepted_clean": self.accepted_clean,
            "dropped_empty": self.dropped_empty,
            "dropped_short": self.dropped_short,
            "dropped_low_alpha": self.dropped_low_alpha,
            "dropped_repetitive": self.dropped_repetitive,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_weight": self.dropped_weight,
        }


class Deduper:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._seen: set[str] = set()

    def seen(self, text: str) -> bool:
        if not self.enabled:
            return False
        digest = hashlib.blake2b(
            text.encode("utf-8", errors="ignore"), digest_size=16
        ).hexdigest()
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False


def normalize_ratios(specs: Sequence[SourceSpec]) -> List[SourceSpec]:
    total = sum(s.ratio for s in specs)
    if total <= 0:
        raise ValueError("Sum of source ratios must be > 0")
    return [SourceSpec(s.name, s.path, s.ratio / total, s.weight) for s in specs]


def compute_target_lines(
    total_lines: int, specs: Sequence[SourceSpec]
) -> Dict[str, int]:
    specs = normalize_ratios(specs)
    targets: Dict[str, int] = {}
    acc = 0
    for i, spec in enumerate(specs):
        if i < len(specs) - 1:
            n = int(total_lines * spec.ratio)
            targets[spec.name] = n
            acc += n
        else:
            targets[spec.name] = total_lines - acc
    return targets


def parse_mapping_args(items: Optional[Sequence[str]], arg_name: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{arg_name} expects NAME=VALUE, got: {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"{arg_name} expects NAME=VALUE, got: {item}")
        mapping[k] = v
    return mapping


def parse_float_mapping(
    items: Optional[Sequence[str]], arg_name: str
) -> Dict[str, float]:
    raw = parse_mapping_args(items, arg_name)
    out: Dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[k] = float(v)
        except ValueError as e:
            raise ValueError(f"{arg_name} expects NAME=float, got: {k}={v}") from e
    return out


def build_source_specs(args: argparse.Namespace) -> List[SourceSpec]:
    if args.config:
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        specs = [
            SourceSpec(
                name=item["name"],
                path=Path(item["path"]),
                ratio=float(item.get("ratio", 1.0)),
                weight=float(item.get("weight", 1.0)),
            )
            for item in data["sources"]
        ]
        return specs

    source_map = parse_mapping_args(args.source, "--source")
    ratio_map = parse_float_mapping(args.ratio, "--ratio")
    weight_map = parse_float_mapping(args.weight, "--weight")

    if not source_map:
        source_map = {
            "zh": "data/clean/novel_clean.txt",
            "en": "data/clean/fineweb_clean.txt",
        }
    if not ratio_map:
        ratio_map = {"zh": 0.75, "en": 0.25}

    specs: List[SourceSpec] = []
    for name, path_str in source_map.items():
        if name not in ratio_map:
            raise ValueError(f"Missing ratio for source: {name}")
        specs.append(
            SourceSpec(
                name=name,
                path=Path(path_str),
                ratio=ratio_map[name],
                weight=weight_map.get(name, 1.0),
            )
        )

    missing = set(ratio_map) - set(source_map)
    if missing:
        raise ValueError(f"Ratio provided for unknown source(s): {sorted(missing)}")

    return specs


def quality_clean(
    line: str,
    *,
    min_line_chars: int,
    max_line_chars: int,
) -> Tuple[Optional[str], str]:
    line = line.strip()
    if not line:
        return None, "empty"

    line = line.replace("\u3000", " ")
    line = line.replace("\ufeff", "")
    line = _WHITESPACE_RE.sub(" ", line)

    if len(line) < min_line_chars:
        return None, "short"

    if len(line) > max_line_chars:
        line = line[:max_line_chars]

    alnum_ratio = sum(ch.isalnum() for ch in line) / max(len(line), 1)
    if alnum_ratio < 0.20:
        return None, "low_alpha"

    if _REPEAT_CHAR_RE.search(line):
        return None, "repetitive"

    # Keep some URLs/HTML for robustness, but drop lines that are mostly markup.
    if (_URL_RE.search(line) or _HTML_TAG_RE.search(line)) and alnum_ratio < 0.35:
        return None, "low_alpha"

    return line, "ok"


def iter_clean_lines(
    path: Path,
    *,
    min_line_chars: int,
    max_line_chars: int,
) -> Iterator[Tuple[Optional[str], str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            yield quality_clean(
                raw,
                min_line_chars=min_line_chars,
                max_line_chars=max_line_chars,
            )


def reservoir_sample_lines(
    spec: SourceSpec,
    *,
    target_lines: int,
    min_line_chars: int,
    max_line_chars: int,
    exact_dedup: bool,
    seed: int,
    progress_every: int,
    fast_dev: bool,
) -> Tuple[List[str], SampleStats]:
    if not spec.path.exists():
        raise FileNotFoundError(
            f"[{spec.name}] missing input file: {spec.path.resolve()}"
        )

    rng = random.Random(seed)
    deduper = Deduper(enabled=exact_dedup)
    sample: List[str] = []
    stats = SampleStats(name=spec.name, target=target_lines)

    print(f"[sample:{spec.name}] reading from  : {spec.path}")
    print(f"[sample:{spec.name}] target lines  : {target_lines:,}")
    print(f"[sample:{spec.name}] source weight : {spec.weight:.4f}")
    print(
        f"[sample:{spec.name}] mode          : {'fast-dev' if fast_dev else 'reservoir'}"
    )

    for cleaned, reason in iter_clean_lines(
        spec.path,
        min_line_chars=min_line_chars,
        max_line_chars=max_line_chars,
    ):
        stats.scanned += 1

        if cleaned is None:
            if reason == "empty":
                stats.dropped_empty += 1
            elif reason == "short":
                stats.dropped_short += 1
            elif reason == "low_alpha":
                stats.dropped_low_alpha += 1
            elif reason == "repetitive":
                stats.dropped_repetitive += 1
            continue

        if deduper.seen(cleaned):
            stats.dropped_duplicate += 1
            continue

        if spec.weight < 1.0 and rng.random() > spec.weight:
            stats.dropped_weight += 1
            continue
        if spec.weight > 1.0:
            # Weight > 1 means always accept, then give additional chances up to cap 3.
            pass

        stats.accepted_clean += 1

        if fast_dev:
            sample.append(cleaned)
            if len(sample) >= target_lines:
                break
        else:
            if len(sample) < target_lines:
                sample.append(cleaned)
            else:
                j = rng.randint(0, stats.accepted_clean - 1)
                if j < target_lines:
                    sample[j] = cleaned

        if spec.weight > 1.0 and len(sample) < target_lines:
            extra_prob = min(spec.weight - 1.0, 2.0)
            if rng.random() < extra_prob:
                sample.append(cleaned)
                if len(sample) > target_lines:
                    sample.pop()

        if stats.scanned % progress_every == 0:
            print(
                f"[sample:{spec.name}] scanned {stats.scanned:,} lines, kept {len(sample):,}"
            )

    stats.kept = len(sample)
    print(f"[sample:{spec.name}] done")
    print(f"[sample:{spec.name}] scanned total : {stats.scanned:,}")
    print(f"[sample:{spec.name}] clean accept  : {stats.accepted_clean:,}")
    print(f"[sample:{spec.name}] written lines : {stats.kept:,}")
    return sample, stats


def write_mixed_sample(
    dst: Path,
    source_samples: Dict[str, List[str]],
    *,
    seed: int,
) -> Dict[str, int]:
    mixed: List[Tuple[str, str]] = []
    for source_name, lines in source_samples.items():
        mixed.extend((source_name, line) for line in lines)

    rng = random.Random(seed)
    rng.shuffle(mixed)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for _, line in mixed:
            f.write(line)
            f.write("\n")

    stats = {k: len(v) for k, v in source_samples.items()}
    total = sum(stats.values())
    print("[mix] done")
    print(f"[mix] total lines : {total:,}")
    for k, v in stats.items():
        pct = 100.0 * v / max(total, 1)
        print(f"[mix] {k:>12} : {v:,} ({pct:.2f}%)")
    print(f"[mix] saved to    : {dst}")
    return stats


def train_tokenizer(
    *,
    input_txt: Path,
    model_prefix: Path,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    input_sentence_size: int,
    max_sentence_length: int,
    num_threads: int,
    control_tokens: Sequence[str],
) -> None:
    print("[spm] start training")
    print(f"[spm] input              : {input_txt}")
    print(f"[spm] model_prefix       : {model_prefix}")
    print(f"[spm] vocab_size         : {vocab_size:,}")
    print(f"[spm] model_type         : {model_type}")
    print(f"[spm] character_coverage : {character_coverage}")
    print(f"[spm] input_sentence_size: {input_sentence_size:,}")
    print(f"[spm] max_sentence_length: {max_sentence_length:,}")
    print(f"[spm] num_threads        : {num_threads}")

    kwargs = dict(
        input=str(input_txt),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
        split_digits=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=False,
        normalization_rule_name="nmt_nfkc",
        remove_extra_whitespaces=True,
        num_threads=num_threads,
        user_defined_symbols=list(control_tokens),
    )

    try:
        spm.SentencePieceTrainer.Train(
            **kwargs,
            train_extremely_large_corpus=True,
        )
    except TypeError:
        print("[spm] train_extremely_large_corpus not supported, fallback.")
        spm.SentencePieceTrainer.Train(**kwargs)

    print("[spm] done")
    print("Saved:", str(model_prefix) + ".model")
    print("Saved:", str(model_prefix) + ".vocab")


def save_manifest(
    *,
    path: Path,
    args: argparse.Namespace,
    specs: Sequence[SourceSpec],
    targets: Dict[str, int],
    mix_stats: Dict[str, int],
    sample_stats: Sequence[SampleStats],
    model_prefix: Path,
) -> None:
    manifest = {
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "input_sentence_size": args.input_sentence_size,
        "total_sample_lines": args.total_sample_lines,
        "sample_txt": str(args.sample_txt),
        "model_prefix": str(model_prefix),
        "control_tokens": args.control_tokens,
        "sources": [
            {
                "name": s.name,
                "path": str(s.path),
                "ratio": s.ratio,
                "weight": s.weight,
                "target_lines": targets[s.name],
                "mixed_lines": mix_stats.get(s.name, 0),
                "stats": next(x.to_dict() for x in sample_stats if x.name == s.name),
            }
            for s in specs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[manifest] saved to     : {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Skull 128k SentencePiece tokenizer"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file with sources",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source mapping NAME=PATH. Example: --source zh=data/clean/novel.txt",
    )
    parser.add_argument(
        "--ratio",
        action="append",
        default=[],
        help="Ratio mapping NAME=FLOAT. Example: --ratio zh=0.75",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Optional source weight NAME=FLOAT. Example: --weight zh_novel=1.5",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-name", type=str, default=DEFAULT_SAMPLE_NAME)
    parser.add_argument("--model-prefix-name", type=str, default=DEFAULT_MODEL_PREFIX)
    parser.add_argument(
        "--manifest-name", type=str, default="tokenizer_manifest_128k.json"
    )
    parser.add_argument(
        "--total-sample-lines", type=int, default=DEFAULT_TOTAL_SAMPLE_LINES
    )
    parser.add_argument(
        "--input-sentence-size", type=int, default=DEFAULT_INPUT_SENTENCE_SIZE
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--model-type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        choices=["bpe", "unigram", "char", "word"],
    )
    parser.add_argument(
        "--character-coverage", type=float, default=DEFAULT_CHARACTER_COVERAGE
    )
    parser.add_argument("--min-line-chars", type=int, default=DEFAULT_MIN_LINE_CHARS)
    parser.add_argument("--max-line-chars", type=int, default=DEFAULT_MAX_LINE_CHARS)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="Stop after target lines instead of full reservoir scan",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Only build the mixed sample file"
    )
    parser.add_argument(
        "--control-token",
        action="append",
        dest="control_tokens",
        default=list(DEFAULT_CONTROL_TOKENS),
        help="Add a user_defined_symbol. Can be repeated.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        specs = build_source_specs(args)
    except ValueError as e:
        parser.error(str(e))
        return

    specs = normalize_ratios(specs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.sample_txt = args.out_dir / args.sample_name
    model_prefix = args.out_dir / args.model_prefix_name
    manifest_path = args.out_dir / args.manifest_name

    print("[plan] Skull tokenizer plan")
    for spec in specs:
        print(
            f"[plan] source={spec.name} path={spec.path} ratio={spec.ratio:.4f} weight={spec.weight:.4f}"
        )
        if not spec.path.exists():
            raise FileNotFoundError(
                f"[{spec.name}] missing input file: {spec.path.resolve()}"
            )

    targets = compute_target_lines(args.total_sample_lines, specs)
    print("[plan] sample targets")
    for name, n in targets.items():
        print(f"[plan] {name:>12} -> {n:,} lines")

    source_samples: Dict[str, List[str]] = {}
    source_stats: List[SampleStats] = []
    for i, spec in enumerate(specs):
        lines, stats = reservoir_sample_lines(
            spec,
            target_lines=targets[spec.name],
            min_line_chars=args.min_line_chars,
            max_line_chars=args.max_line_chars,
            exact_dedup=not args.no_dedup,
            seed=args.seed + i * 1009,
            progress_every=args.progress_every,
            fast_dev=args.fast_dev,
        )
        source_samples[spec.name] = lines
        source_stats.append(stats)

    mix_stats = write_mixed_sample(args.sample_txt, source_samples, seed=args.seed)

    save_manifest(
        path=manifest_path,
        args=args,
        specs=specs,
        targets=targets,
        mix_stats=mix_stats,
        sample_stats=source_stats,
        model_prefix=model_prefix,
    )

    if args.skip_train:
        print("[spm] skip training requested")
        return

    train_tokenizer(
        input_txt=args.sample_txt,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        max_sentence_length=args.max_line_chars,
        num_threads=args.num_threads,
        control_tokens=args.control_tokens,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[exit] interrupted by user", file=sys.stderr)
        raise SystemExit(130)
