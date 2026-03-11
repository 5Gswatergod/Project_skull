from pathlib import Path
import hashlib
import html
import re
import unicodedata

INPUT_DIR = Path("data/corpus/raw/novel")
OUTPUT_TXT = Path("data/clean/novel_clean.txt")

MIN_LEN = 5
MAX_LEN = 2000
DEDUP = True

MULTISPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# 日文
JP_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]")

# 只保留：中文、英文、數字、空白、常用中英文標點
VALID_RE = re.compile(
    r"[^\u4e00-\u9fff\u3400-\u4dbfA-Za-z0-9\s。，、！？；：「」『』（）《》〈〉—…,.!?;:'\"()\[\]\-]"
)

MOJIBAKE_CHARS = set("�")
REPEAT_CHAR_RE = re.compile(r"(.)\1{7,}")


def contains_japanese(text: str) -> bool:
    return bool(JP_RE.search(text))


def looks_garbled(text: str) -> bool:
    if not text:
        return True

    if any(c in MOJIBAKE_CHARS for c in text):
        return True

    weird = 0
    for c in text:
        cat = unicodedata.category(c)
        if cat.startswith("C") and c not in ("\n", "\t", " "):
            weird += 1

    if weird / max(len(text), 1) > 0.05:
        return True

    if REPEAT_CHAR_RE.search(text):
        return True

    return False


def normalize_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    line = html.unescape(line)
    line = unicodedata.normalize("NFKC", line)
    line = CONTROL_RE.sub(" ", line)
    line = line.replace("\u3000", " ")

    line = HTML_RE.sub(" ", line)
    line = URL_RE.sub(" ", line)
    line = MULTISPACE_RE.sub(" ", line).strip()

    if not line:
        return ""

    # 先砍日文
    if contains_japanese(line):
        return ""

    # 只留中英數與標點
    line = VALID_RE.sub(" ", line)
    line = MULTISPACE_RE.sub(" ", line).strip()

    if not line:
        return ""

    if len(line) < MIN_LEN or len(line) > MAX_LEN:
        return ""

    if looks_garbled(line):
        return ""

    return line


def iter_txt_files(input_dir: Path):
    for path in sorted(input_dir.glob("*.txt")):
        if path.is_file():
            yield path


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input dir: {INPUT_DIR.resolve()}")

    files = list(iter_txt_files(INPUT_DIR))
    if not files:
        raise FileNotFoundError(f"No .txt files found under: {INPUT_DIR.resolve()}")

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    total_dup = 0
    total_skipped = 0

    seen = set()

    with open(OUTPUT_TXT, "w", encoding="utf-8", newline="\n") as out:
        print("Merging files:")
        for f in files:
            print(" ", f.name)

        for f in files:
            file_in = 0
            file_out = 0
            file_dup = 0
            file_skipped = 0

            with open(f, "r", encoding="utf-8", errors="ignore") as src:
                for raw in src:
                    file_in += 1
                    total_in += 1

                    # line = normalize_line(raw)
                    line = raw
                    if not line:
                        file_skipped += 1
                        total_skipped += 1
                        continue

                    if DEDUP:
                        h = hashlib.blake2b(
                            line.encode("utf-8"), digest_size=16
                        ).digest()
                        if h in seen:
                            file_dup += 1
                            total_dup += 1
                            continue
                        seen.add(h)

                    out.write(line + "\n")
                    file_out += 1
                    total_out += 1

            print(
                f"[done] {f.name} | in={file_in:,} kept={file_out:,} "
                f"dup={file_dup:,} skipped={file_skipped:,}"
            )

    print("\n==== MERGE DONE ====")
    print(f"output   : {OUTPUT_TXT}")
    print(f"input    : {total_in:,} lines")
    print(f"kept     : {total_out:,} lines")
    print(f"deduped  : {total_dup:,} lines")
    print(f"skipped  : {total_skipped:,} lines")


if __name__ == "__main__":
    main()
