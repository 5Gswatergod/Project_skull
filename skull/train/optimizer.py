from __future__ import annotations

import inspect

import torch


def split_weight_decay_params(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Standard GPT rule: matrix weights get decay, vectors/scalars do not.
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    return decay, no_decay


def build_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 0.1))
    betas = tuple(cfg.get("betas", [0.9, 0.95]))
    eps = float(cfg.get("eps", 1e-8))

    decay, no_decay = split_weight_decay_params(model)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if optimizer_name == "adamw":
        kwargs = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
        }
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            kwargs["fused"] = bool(
                cfg.get(
                    "fused_adamw",
                    torch.cuda.is_available(),
                )
            )
        return torch.optim.AdamW(param_groups, **kwargs)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
