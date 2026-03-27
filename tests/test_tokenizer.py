from pathlib import Path

import pytest

from skull.tokenization import load_tokenizer


def test_load_tokenizer_if_model_exists():
    model_path = Path("data/tokenizer/skull_zh_en_128k.model")
    if not model_path.exists():
        pytest.skip("SentencePiece model not found")

    tokenizer = load_tokenizer(model_path)
    assert tokenizer.vocab_size > 0

    ids = tokenizer.encode("你好 Project Skull", add_bos=True, add_eos=True)
    assert isinstance(ids, list)
    assert len(ids) >= 2

    text = tokenizer.decode(ids)
    assert isinstance(text, str)
    assert len(text) > 0
