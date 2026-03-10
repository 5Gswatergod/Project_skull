from __future__ import annotations

from skull.eval.generation import generate_text
from skull.eval.perplexity import evaluate_perplexity_from_cfg
from skull.eval.prompts import DEFAULT_PROMPTS


def run_basic_benchmark(model, tokenizer, eval_cfg: dict, device):
    metrics = evaluate_perplexity_from_cfg(model=model, cfg=eval_cfg, device=device)
    samples = [
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(eval_cfg.get("sample_max_new_tokens", 64)),
            temperature=float(eval_cfg.get("temperature", 1.0)),
            top_k=eval_cfg.get("top_k", None),
            device=device,
        )
        for prompt in eval_cfg.get("sample_prompts", DEFAULT_PROMPTS)
    ]
    return {
        "metrics": metrics,
        "samples": samples,
    }
