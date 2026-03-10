import torch

from skull.model import GPT, GPTConfig


def test_model_forward():
    cfg = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        bias=False,
        norm="layernorm",
        pos_encoding="absolute",
        mlp_type="gelu",
        use_checkpointing=False,
    )
    model = GPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x)

    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == (2, 8, cfg.vocab_size)


def test_model_forward_with_targets():
    cfg = GPTConfig(
        vocab_size=128,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        bias=False,
        norm="rmsnorm",
        pos_encoding="rope",
        mlp_type="swiglu",
        use_checkpointing=False,
    )
    model = GPT(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x, targets=y)

    assert "logits" in out
    assert "loss" in out
    assert out["logits"].shape == (2, 8, cfg.vocab_size)
    assert out["loss"].ndim == 0
