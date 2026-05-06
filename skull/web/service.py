from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from skull.web.data import collect_dashboard_state
from skull.web.jobs import (
    build_eval_command,
    build_pytest_command,
    build_sample_command,
    build_train_command,
    current_time_iso,
    delete_job,
    find_job,
    load_jobs,
    read_log_tail,
    request_stop,
    start_job,
)


TrainMode = Literal["auto", "pretrain", "cpt", "sft"]


def default_repo_root() -> Path:
    configured = os.environ.get("SKULL_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    path = (
        default_repo_root()
        if repo_root is None or str(repo_root).strip() == ""
        else Path(str(repo_root)).expanduser().resolve()
    )
    if not path.exists():
        raise ValueError(f"Repo root does not exist: {path}")
    return path


def _repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _ensure_path_exists(repo_root: Path, path_value: str, label: str) -> Path:
    path = _repo_path(repo_root, path_value)
    if not path_value or not path.exists():
        raise ValueError(f"{label} not found: {path}")
    return path


def _train_configs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in state["configs"] if item.get("kind") == "train"]


def _eval_configs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in state["configs"] if item.get("kind") == "eval"]


def _checkpoint_paths(state: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for run in state["runs"]:
        for checkpoint in run.get("checkpoints", []):
            path = checkpoint.get("relative_path")
            if path:
                paths.append(str(path))
    return sorted(dict.fromkeys(paths))


def _train_config_record(state: dict[str, Any], config_path: str) -> dict[str, Any] | None:
    return next(
        (
            config
            for config in _train_configs(state)
            if config.get("relative_path") == config_path
        ),
        None,
    )


def infer_train_mode(
    config: dict[str, Any] | None,
    config_path: str,
    requested: TrainMode = "auto",
) -> str:
    if requested != "auto":
        return requested

    content = config.get("content", {}) if config else {}
    path_text = config_path.replace("\\", "/").lower()
    run_dir = str(content.get("run_dir", "")).replace("\\", "/").lower()

    if "train_jsonl" in content or "/sft/" in run_dir or "sft" in path_text:
        return "sft"
    if "base_ckpt" in content or "/cpt/" in run_dir or "cpt" in path_text:
        return "cpt"
    return "pretrain"


def build_dashboard_payload(repo_root: Path) -> dict[str, Any]:
    state = collect_dashboard_state(repo_root)
    jobs = load_jobs(repo_root)
    return {
        "repo_root": str(repo_root),
        "generated_at": current_time_iso(),
        "state": state,
        "jobs": jobs,
        "launchpad": {
            "train_configs": [
                {
                    "path": item.get("relative_path"),
                    "run_name": item.get("run_name"),
                    "run_dir": item.get("run_dir"),
                    "device": item.get("device"),
                    "max_steps": item.get("max_steps"),
                    "tokenizer_exists": item.get("tokenizer_exists"),
                    "model_config_exists": item.get("model_config_exists"),
                }
                for item in _train_configs(state)
            ],
            "eval_configs": [
                {
                    "path": item.get("relative_path"),
                    "eval_source_count": item.get("eval_source_count"),
                }
                for item in _eval_configs(state)
            ],
            "checkpoints": _checkpoint_paths(state),
        },
    }


def get_job_or_raise(repo_root: Path, job_id: str) -> dict[str, Any]:
    job = find_job(repo_root, job_id)
    if job is None:
        raise KeyError(f"Unknown job id: {job_id}")
    return job


def get_job_log(repo_root: Path, job_id: str, *, max_chars: int = 20000) -> dict[str, Any]:
    job = get_job_or_raise(repo_root, job_id)
    return {
        "job": job,
        "log": read_log_tail(job, max_chars=max(1000, int(max_chars))),
        "generated_at": current_time_iso(),
    }


def launch_train_job(
    repo_root: Path,
    *,
    config_path: str,
    requested_mode: TrainMode = "auto",
    use_accelerate: bool = False,
    num_processes: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    state = collect_dashboard_state(repo_root)
    config = _train_config_record(state, config_path)
    if config is None:
        raise ValueError(f"Unknown train config: {config_path}")

    _ensure_path_exists(repo_root, config_path, "Config")
    mode = infer_train_mode(config, config_path, requested=requested_mode)

    if num_processes is not None and int(num_processes) < 1:
        raise ValueError("num_processes must be at least 1.")

    job_label = (label or "").strip() or f"train:{mode}:{Path(config_path).stem}"
    return start_job(
        repo_root,
        job_type=f"train:{mode}",
        label=job_label,
        command=build_train_command(
            mode,
            config_path,
            use_accelerate=use_accelerate,
            num_processes=int(num_processes) if num_processes is not None else None,
        ),
        metadata={
            "config": config_path,
            "mode": mode,
            "accelerate": bool(use_accelerate),
        },
    )


def launch_eval_job(
    repo_root: Path,
    *,
    config_path: str,
    checkpoint_path: str,
    print_json: bool = True,
    label: str | None = None,
) -> dict[str, Any]:
    _ensure_path_exists(repo_root, config_path, "Config")
    _ensure_path_exists(repo_root, checkpoint_path, "Checkpoint")

    job_label = (label or "").strip() or f"eval:{Path(checkpoint_path).stem}"
    return start_job(
        repo_root,
        job_type="eval",
        label=job_label,
        command=build_eval_command(
            config_path,
            checkpoint_path,
            print_json=print_json,
        ),
        metadata={
            "config": config_path,
            "checkpoint": checkpoint_path,
        },
    )


def launch_sample_job(
    repo_root: Path,
    *,
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    _ensure_path_exists(repo_root, config_path, "Config")
    _ensure_path_exists(repo_root, checkpoint_path, "Checkpoint")
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be at least 1.")
    if float(temperature) <= 0:
        raise ValueError("temperature must be greater than 0.")
    if top_k is not None and int(top_k) < 1:
        raise ValueError("top_k must be at least 1.")

    job_label = (label or "").strip() or f"sample:{Path(checkpoint_path).stem}"
    return start_job(
        repo_root,
        job_type="sample",
        label=job_label,
        command=build_sample_command(
            config_path,
            checkpoint_path,
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k is not None else None,
        ),
        metadata={
            "config": config_path,
            "checkpoint": checkpoint_path,
        },
    )


def launch_test_job(
    repo_root: Path,
    *,
    targets: list[str] | None = None,
    extra_args: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    cleaned_targets = [str(target).strip() for target in (targets or []) if str(target).strip()]
    cleaned_args = [str(arg).strip() for arg in (extra_args or []) if str(arg).strip()]

    job_label = (label or "").strip() or "tests:pytest"
    return start_job(
        repo_root,
        job_type="test",
        label=job_label,
        command=build_pytest_command(cleaned_targets or ["tests"], cleaned_args),
        metadata={"targets": cleaned_targets or ["tests"]},
    )


def stop_job(repo_root: Path, job_id: str) -> dict[str, Any]:
    get_job_or_raise(repo_root, job_id)
    return request_stop(repo_root, job_id)


def delete_finished_job(
    repo_root: Path,
    job_id: str,
    *,
    delete_log_too: bool = False,
) -> dict[str, Any]:
    get_job_or_raise(repo_root, job_id)
    return delete_job(repo_root, job_id, delete_log=delete_log_too)
