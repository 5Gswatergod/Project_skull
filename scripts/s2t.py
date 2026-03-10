from pathlib import Path
from opencc import OpenCC
import chardet
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ===== 固定路徑 =====
INPUT_DIR = Path("data/corpus/raw/novel")
OUTPUT_DIR = Path("data/corpus/staging/novel")

EXTENSIONS = {".txt", ".md"}

cc = OpenCC("s2t")


def detect_encoding(file_path):
    raw = file_path.read_bytes()
    result = chardet.detect(raw)
    return result["encoding"]


def read_text_auto(file_path):
    raw = file_path.read_bytes()
    encoding = detect_encoding(file_path)

    if encoding is None:
        raise ValueError("Unknown encoding")

    return raw.decode(encoding, errors="ignore")


def convert_path(relative_path):
    """
    將整個路徑轉成繁體
    """
    parts = relative_path.parts
    new_parts = [cc.convert(p) for p in parts]
    return Path(*new_parts)


def process_file(src):

    try:
        text = read_text_auto(src)
    except Exception:
        return "skip"

    converted = cc.convert(text)

    relative = src.relative_to(INPUT_DIR)

    # 路徑 + 檔名轉繁體
    new_relative = convert_path(relative)

    dst = OUTPUT_DIR / new_relative
    dst.parent.mkdir(parents=True, exist_ok=True)

    dst.write_text(converted, encoding="utf-8")

    return "ok"


def collect_files():

    files = []

    for file in INPUT_DIR.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in EXTENSIONS:
            continue

        files.append(file)

    return files


def main():

    if not INPUT_DIR.exists():
        print(f"找不到資料夾: {INPUT_DIR}")
        return

    files = collect_files()

    print(f"找到 {len(files)} 個檔案")

    ok = 0
    skip = 0

    with Pool(cpu_count()) as pool:

        for result in tqdm(pool.imap_unordered(process_file, files), total=len(files)):

            if result == "ok":
                ok += 1
            else:
                skip += 1

    print("\n====== 完成 ======")
    print("成功:", ok)
    print("跳過:", skip)
    print("輸出:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
