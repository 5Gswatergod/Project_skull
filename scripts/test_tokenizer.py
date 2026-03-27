from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterable, List, Sequence

import sentencepiece as spm

ROOT = Path('.')
DEFAULT_MODEL = ROOT / 'data' / 'tokenizer' / 'skull_zh_en_128k_bpe.model'
DEFAULT_INPUTS = [
    '今天天氣很好，我想用 Skull 測試中文 tokenizer。',
    'Skull tokenizer should handle mixed English and 中文 properly.',
    '大型語言模型正在學習 subword tokenization 與數字 2026 的表示。',
    '<|system|> You are Skull. <|user|> 請解釋 tokenizer 是什麼。',
]


def iter_lines_from_file(path: Path, limit: int | None = None) -> Iterable[str]:
    count = 0
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield line
            count += 1
            if limit is not None and count >= limit:
                break



def shorten(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + '...'



def load_texts(texts: Sequence[str], input_file: Path | None, limit: int | None) -> List[str]:
    loaded = list(texts)
    if input_file is not None:
        loaded.extend(iter_lines_from_file(input_file, limit=limit))
    return loaded



def roundtrip_ok(sp: spm.SentencePieceProcessor, text: str) -> bool:
    ids = sp.encode(text, out_type=int)
    decoded = sp.decode(ids)
    return decoded == text



def inspect_one(sp: spm.SentencePieceProcessor, text: str, index: int) -> tuple[int, int]:
    pieces = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)
    decoded = sp.decode(ids)

    print(f'[{index}] text      : {shorten(text)}')
    print(f'[{index}] chars     : {len(text)}')
    print(f'[{index}] tokens    : {len(ids)}')
    print(f'[{index}] chars/tok : {len(text) / max(len(ids), 1):.3f}')
    print(f'[{index}] pieces    : {pieces[:80]}')
    print(f'[{index}] ids       : {ids[:80]}')
    print(f'[{index}] decoded== : {decoded == text}')
    if decoded != text:
        print(f'[{index}] decoded   : {shorten(decoded)}')
    print('-' * 80)

    return len(text), len(ids)



def print_vocab_preview(sp: spm.SentencePieceProcessor, count: int) -> None:
    print('[vocab] preview')
    for i in range(min(count, sp.get_piece_size())):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        print(f'[vocab] {i:>6} | {piece!r} | score={score:.6f}')
    print('-' * 80)



def print_special_ids(sp: spm.SentencePieceProcessor) -> None:
    print('[special] ids')
    print(f'[special] unk_id = {sp.unk_id()}')
    print(f'[special] bos_id = {sp.bos_id()}')
    print(f'[special] eos_id = {sp.eos_id()}')
    print(f'[special] pad_id = {sp.pad_id()}')
    for token in [
        '<|system|>',
        '<|user|>',
        '<|assistant|>',
        '<|tool|>',
        '<|observation|>',
        '<|end|>',
        '<|text|>',
        '<|code|>',
    ]:
        print(f'[special] piece_to_id({token!r}) = {sp.piece_to_id(token)}')
    print('-' * 80)



def summary(sp: spm.SentencePieceProcessor, texts: Sequence[str]) -> None:
    char_counts: List[int] = []
    token_counts: List[int] = []
    roundtrip_pass = 0

    for i, text in enumerate(texts, start=1):
        chars, toks = inspect_one(sp, text, i)
        char_counts.append(chars)
        token_counts.append(toks)
        if roundtrip_ok(sp, text):
            roundtrip_pass += 1

    total_chars = sum(char_counts)
    total_tokens = sum(token_counts)
    ratio = total_chars / max(total_tokens, 1)

    print('[summary]')
    print(f'[summary] samples             : {len(texts)}')
    print(f'[summary] vocab_size          : {sp.get_piece_size():,}')
    print(f'[summary] total_chars         : {total_chars:,}')
    print(f'[summary] total_tokens        : {total_tokens:,}')
    print(f'[summary] avg_chars_per_token : {ratio:.3f}')
    print(f'[summary] avg_tokens_per_text : {statistics.mean(token_counts):.3f}')
    print(f'[summary] median_tokens       : {statistics.median(token_counts):.3f}')
    print(f'[summary] max_tokens          : {max(token_counts)}')
    print(f'[summary] min_tokens          : {min(token_counts)}')
    print(f'[summary] roundtrip_pass      : {roundtrip_pass}/{len(texts)}')



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Inspect and test a SentencePiece tokenizer for Project Skull.'
    )
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        '--text',
        action='append',
        default=[],
        help='Repeatable inline test text. Example: --text "你好 world"',
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        default=None,
        help='Optional text file with one sample per line.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Max non-empty lines to read from --input-file.',
    )
    parser.add_argument(
        '--show-vocab',
        type=int,
        default=0,
        help='Show the first N vocab pieces.',
    )
    return parser



def main() -> None:
    args = build_argparser().parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f'missing tokenizer model: {args.model.resolve()}')

    sp = spm.SentencePieceProcessor()
    loaded = sp.load(str(args.model))
    if not loaded:
        raise RuntimeError(f'failed to load tokenizer model: {args.model}')

    texts = load_texts(
        texts=args.text if args.text else DEFAULT_INPUTS,
        input_file=args.input_file,
        limit=args.limit,
    )
    if not texts:
        raise ValueError('no test texts found; provide --text or --input-file')

    print(f'[model] path       : {args.model}')
    print(f'[model] vocab_size : {sp.get_piece_size():,}')
    print('-' * 80)

    print_special_ids(sp)

    if args.show_vocab > 0:
        print_vocab_preview(sp, args.show_vocab)

    summary(sp, texts)


if __name__ == '__main__':
    main()
