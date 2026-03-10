from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t\u3000]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_REPEAT_PUNCT_RE = re.compile(r"([!！?？。．\.，,、；;：:])\1{4,}")
_CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")


@dataclass
class CleaningConfig:
    strip_html: bool = True
    strip_urls: bool = True
    unescape_html_entities: bool = True
    unicode_normalize: str = "NFKC"
    remove_control_chars: bool = True
    collapse_spaces: bool = True
    collapse_newlines: bool = True
    strip_edges: bool = True
    min_chars: int = 10
    max_chars: int = 20000
    min_cjk_ratio: float = 0.0
    min_alpha_ratio: float = 0.0
    max_digit_ratio: float = 0.6
    max_punct_ratio: float = 0.5
    drop_if_empty: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _punct_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct_count = sum(1 for ch in text if unicodedata.category(ch).startswith("P"))
    return punct_count / len(text)


def _char_profile(text: str) -> dict:
    total = len(text)
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    digit = len(_DIGIT_RE.findall(text))
    punct_ratio = _punct_ratio(text)
    return {
        "total": total,
        "cjk": cjk,
        "latin": latin,
        "digit": digit,
        "cjk_ratio": _safe_ratio(cjk, total),
        "alpha_ratio": _safe_ratio(cjk + latin, total),
        "digit_ratio": _safe_ratio(digit, total),
        "punct_ratio": punct_ratio,
    }


def clean_text(text: str, config: Optional[CleaningConfig] = None) -> str:
    cfg = config or CleaningConfig()
    if text is None:
        return ""

    if cfg.unescape_html_entities:
        text = html.unescape(text)

    if cfg.unicode_normalize:
        text = unicodedata.normalize(cfg.unicode_normalize, text)

    if cfg.strip_urls:
        text = _URL_RE.sub(" ", text)

    if cfg.strip_html:
        text = _HTML_TAG_RE.sub(" ", text)

    if cfg.remove_control_chars:
        text = _CONTROL_RE.sub("", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _REPEAT_PUNCT_RE.sub(r"\1\1\1", text)

    if cfg.collapse_spaces:
        text = _MULTI_SPACE_RE.sub(" ", text)

    if cfg.collapse_newlines:
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    if cfg.strip_edges:
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = text.strip()

    return text


def _passes_filters(text: str, config: CleaningConfig) -> bool:
    if config.drop_if_empty and not text:
        return False

    n = len(text)
    if n < config.min_chars:
        return False
    if config.max_chars > 0 and n > config.max_chars:
        return False

    stats = _char_profile(text)

    if stats["cjk_ratio"] < config.min_cjk_ratio:
        return False
    if stats["alpha_ratio"] < config.min_alpha_ratio:
        return False
    if stats["digit_ratio"] > config.max_digit_ratio:
        return False
    if stats["punct_ratio"] > config.max_punct_ratio:
        return False

    return True


def clean_lines(
    lines: Iterable[str],
    config: Optional[CleaningConfig] = None,
) -> Iterator[str]:
    cfg = config or CleaningConfig()
    for line in lines:
        line = clean_text(line, cfg)
        if _passes_filters(line, cfg):
            yield line


def iter_clean_file(
    path: str | Path,
    config: Optional[CleaningConfig] = None,
    encoding: str = "utf-8",
) -> Iterator[str]:
    with open(path, "r", encoding=encoding) as f:
        yield from clean_lines(f, config=config)


def write_clean_file(
    input_path: str | Path,
    output_path: str | Path,
    config: Optional[CleaningConfig] = None,
    encoding: str = "utf-8",
    stats_path: Optional[str | Path] = None,
) -> dict:
    cfg = config or CleaningConfig()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    kept_lines = 0
    dropped_lines = 0
    kept_chars = 0

    with open(input_path, "r", encoding=encoding) as fin, open(
        output_path, "w", encoding=encoding
    ) as fout:
        for raw in fin:
            total_lines += 1
            cleaned = clean_text(raw, cfg)
            if not _passes_filters(cleaned, cfg):
                dropped_lines += 1
                continue
            fout.write(cleaned + "\n")
            kept_lines += 1
            kept_chars += len(cleaned)

    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_lines": total_lines,
        "kept_lines": kept_lines,
        "dropped_lines": dropped_lines,
        "kept_chars": kept_chars,
        "config": cfg.to_dict(),
    }

    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats
