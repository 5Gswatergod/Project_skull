from __future__ import annotations

from skull.train.checkpointing import load_checkpoint, resolve_checkpoint_path
from skull.train.trainer_pretrain import PretrainTrainer


class CPTTrainer(PretrainTrainer):
    def __init__(self, cfg: dict, model, tokenizer=None) -> None:
        cfg = dict(cfg)
        cfg.setdefault("run_name", "skull_cpt")
        cfg.setdefault("run_dir", f"runs/cpt/{cfg['run_name']}")

        base_ckpt = cfg.get("base_ckpt")
        if base_ckpt:
            ckpt_path = resolve_checkpoint_path(base_ckpt)
            if ckpt_path is None:
                raise FileNotFoundError(f"base_ckpt does not exist: {base_ckpt}")
            load_checkpoint(
                ckpt_path,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                map_location="cpu",
                strict=True,
            )
            print(f"[cpt] loaded base checkpoint: {ckpt_path}")

        super().__init__(cfg=cfg, model=model, tokenizer=tokenizer)
