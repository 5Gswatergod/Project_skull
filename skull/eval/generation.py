from __future__ import annotations

import torch


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device | None = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=getattr(tokenizer, "eos_id", None),
    )
    return tokenizer.decode(y[0].tolist(), skip_special_tokens=False)
