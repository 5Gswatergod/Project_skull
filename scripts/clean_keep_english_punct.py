#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

# 保留英文字母、數字、空白、常見英文標點
ALLOWED_RE = re.compile(
    r"[^A-Za-z0-9\s\.\,\!\?\:\;\"\'\-\(\)\[\]\{\}/\\@#\$%\^&\*_+=<>\|~`]"
)

MULTI_SPACE_RE = re.compile(r"[ \t]+")


def clean_line(line: str, keep_digits: bool = True) -> str:
    if not keep_digits:
        allowed_re = re.compile(
            r"[^A-Za-z\s\.\,\!\?\:\;\"\'\-\(\)\[\]\{\}/\\@#\$%\^&\*_+=<>\|~`]"
        )
    else:
        allowed_re = ALLOWED_RE

    line = line.replace("\r", "")
    line = allowed_re.sub("", line)
    line = MULTI_SPACE_RE.sub(" ", line).strip()
    return line


def iter_txt_files(input_path: Path, recursive: bool = True) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    pattern = "**/*.txt" if recursive else "*.txt"
    yield from input_path.glob(pattern)


def process_file_stream(
    src: Path,
    dst: Path,
    keep_digits: bool = True,
    encoding: str = "utf-8",
    min_line_len: int = 0,
) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw_chars = 0
    clean_chars = 0
    empty_count = 0

    with (
        src.open("r", encoding=encoding, errors="ignore") as fin,
        dst.open("w", encoding="utf-8") as fout,
    ):

        for line in fin:
            raw_chars += len(line)

            cleaned = clean_line(line, keep_digits=keep_digits)

            if len(cleaned) < min_line_len:
                continue

            if cleaned:
                fout.write(cleaned + "\n")
                clean_chars += len(cleaned) + 1
                empty_count = 0
            else:
                # 避免連續很多空行
                empty_count += 1
                if empty_count <= 1:
                    fout.write("\n")
                    clean_chars += 1

    return raw_chars, clean_chars


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory-efficient English text cleaner"
    )
    parser.add_argument("--input", type=str, required=True, help="輸入檔案或資料夾")
    parser.add_argument("--output", type=str, required=True, help="輸出檔案或資料夾")
    parser.add_argument(
        "--no-recursive", action="store_true", help="資料夾模式下不遞迴"
    )
    parser.add_argument("--no-digits", action="store_true", help="不保留數字")
    parser.add_argument("--encoding", type=str, default="utf-8", help="輸入編碼")
    parser.add_argument("--min-line-len", type=int, default=0, help="過短行直接丟掉")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    recursive = not args.no_recursive
    keep_digits = not args.no_digits

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    total_raw = 0
    total_clean = 0
    processed = 0

    if input_path.is_file():
        dst = output_path / input_path.name if output_path.is_dir() else output_path
        raw_len, clean_len = process_file_stream(
            input_path,
            dst,
            keep_digits=keep_digits,
            encoding=args.encoding,
            min_line_len=args.min_line_len,
        )
        total_raw += raw_len
        total_clean += clean_len
        processed += 1
        print(f"[OK] {input_path} -> {dst}")
    else:
        for src in iter_txt_files(input_path, recursive=recursive):
            rel = src.relative_to(input_path)
            dst = output_path / rel
            raw_len, clean_len = process_file_stream(
                src,
                dst,
                keep_digits=keep_digits,
                encoding=args.encoding,
                min_line_len=args.min_line_len,
            )
            total_raw += raw_len
            total_clean += clean_len
            processed += 1
            print(f"[OK] {src} -> {dst}")

    print("\n=== Summary ===")
    print(f"Processed files : {processed}")
    print(f"Raw chars       : {total_raw}")
    print(f"Clean chars     : {total_clean}")
    if total_raw > 0:
        print(f"Kept ratio      : {total_clean / total_raw:.4f}")


if __name__ == "__main__":
    main()
