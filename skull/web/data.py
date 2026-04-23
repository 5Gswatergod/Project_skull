from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _as_path(root: str | Path) -> Path:
    return Path(root).resolve()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {
                    "line_number": line_number,
                    "decode_error": True,
                    "raw_preview": line[:200],
                }
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _format_bytes(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"

    units = ["KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{size_bytes} B"


def _timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "absolute_path": str(path.resolve()),
        "relative_path": _relative_path(path, root),
        "size_bytes": int(stat.st_size),
        "size_human": _format_bytes(int(stat.st_size)),
        "modified_at": _timestamp(path),
    }


def _dir_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "absolute_path": str(path.resolve()),
        "relative_path": _relative_path(path, root),
        "modified_at": _timestamp(path),
    }


def _safe_sum_file_sizes(paths: list[Path]) -> int:
    return sum(int(path.stat().st_size) for path in paths if path.exists())


def _preview_text(path: Path, *, max_chars: int = 1600) -> str:
    if not path.exists():
        return ""

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def _scan_configs(root: Path) -> list[dict[str, Any]]:
    configs_root = root / "configs"
    if not configs_root.exists():
        return []

    config_paths = sorted(
        set(configs_root.rglob("*.yaml")) | set(configs_root.rglob("*.yml"))
    )
    records: list[dict[str, Any]] = []

    for path in config_paths:
        content = _load_yaml(path)
        parts = path.relative_to(configs_root).parts
        kind = parts[0] if parts else "other"

        record = _file_record(path, root)
        record.update(
            {
                "kind": kind,
                "keys": sorted(content.keys()),
                "content": content,
            }
        )

        if kind == "train":
            train_sources = content.get("train_sources") or []
            val_sources = content.get("val_sources") or []
            tokenizer_model = content.get("tokenizer_model")
            model_config = content.get("model_config")
            record.update(
                {
                    "run_name": content.get("run_name"),
                    "run_dir": content.get("run_dir"),
                    "device": content.get("device"),
                    "batch_size": content.get("batch_size"),
                    "block_size": content.get("block_size"),
                    "max_steps": content.get("max_steps"),
                    "train_source_count": len(train_sources),
                    "val_source_count": len(val_sources),
                    "train_shard_count": sum(
                        len(source.get("paths", []))
                        for source in train_sources
                        if isinstance(source, dict)
                    ),
                    "val_shard_count": sum(
                        len(source.get("paths", []))
                        for source in val_sources
                        if isinstance(source, dict)
                    ),
                    "tokenizer_model": tokenizer_model,
                    "tokenizer_exists": bool(
                        tokenizer_model and (root / tokenizer_model).exists()
                    ),
                    "model_config": model_config,
                    "model_config_exists": bool(
                        model_config and (root / model_config).exists()
                    ),
                }
            )
        elif kind == "model":
            record.update(
                {
                    "block_size": content.get("block_size"),
                    "vocab_size": content.get("vocab_size"),
                    "n_layer": content.get("n_layer"),
                    "n_head": content.get("n_head"),
                    "n_embd": content.get("n_embd"),
                    "pos_encoding": content.get("pos_encoding"),
                    "mlp_type": content.get("mlp_type"),
                }
            )
        elif kind == "eval":
            record.update(
                {
                    "eval_source_count": len(content.get("eval_sources") or []),
                }
            )

        records.append(record)

    return records


def _scan_corpora_registry(root: Path) -> dict[str, Any]:
    registry_path = root / "configs" / "data" / "corpora.yaml"
    content = _load_yaml(registry_path)
    sources = content.get("sources") or []
    mixes = content.get("mixes") or {}

    source_records = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        clean_output = source.get("clean_output") or {}
        bin_output = source.get("bin_output") or {}
        source_records.append(
            {
                "name": source.get("name"),
                "enabled": bool(source.get("enabled", True)),
                "format": source.get("format"),
                "domain": source.get("domain"),
                "lang": source.get("lang"),
                "tokenizer_sampling_weight": source.get(
                    "tokenizer_sampling_weight", 0.0
                ),
                "pretrain_weight": source.get("pretrain_weight", 0.0),
                "cpt_weight": source.get("cpt_weight", 0.0),
                "clean_text_path": clean_output.get("clean_text_path"),
                "bin_output_dir": bin_output.get("output_dir"),
            }
        )

    mix_records = []
    for name, mix in mixes.items():
        if not isinstance(mix, dict):
            continue
        mix_records.append(
            {
                "name": name,
                "description": mix.get("description", ""),
                "source_count": len(mix.get("sources") or []),
                "val_source_count": len(mix.get("val_sources") or []),
            }
        )

    return {
        "path": str(registry_path.resolve()),
        "exists": registry_path.exists(),
        "source_count": len(source_records),
        "mix_count": len(mix_records),
        "sources": source_records,
        "mixes": mix_records,
        "content": content,
    }


def _scan_data_assets(root: Path) -> dict[str, Any]:
    data_root = root / "data"
    tokenizer_root = data_root / "tokenizer"
    clean_root = data_root / "clean"
    bins_root = data_root / "bins"
    manifest_root = data_root / "manifest"

    tokenizer_files = []
    if tokenizer_root.exists():
        tokenizer_files = [
            _file_record(path, root)
            for path in sorted(tokenizer_root.iterdir())
            if path.is_file()
        ]

    clean_files = []
    if clean_root.exists():
        clean_files = [
            _file_record(path, root)
            for path in sorted(clean_root.iterdir())
            if path.is_file()
        ]

    manifest_files = []
    if manifest_root.exists():
        manifest_files = [
            _file_record(path, root)
            for path in sorted(manifest_root.rglob("*"))
            if path.is_file()
        ]

    bin_sets: list[dict[str, Any]] = []
    if bins_root.exists():
        for directory in sorted(path for path in bins_root.iterdir() if path.is_dir()):
            shard_files = sorted(directory.glob("*.bin"))
            train_shards = sorted(directory.glob("train_*.bin"))
            val_shards = sorted(directory.glob("val_*.bin"))
            meta_path = directory / "meta.json"
            meta = _load_json(meta_path)
            total_size_bytes = _safe_sum_file_sizes(shard_files)

            record = _dir_record(directory, root)
            record.update(
                {
                    "meta_exists": meta_path.exists(),
                    "meta": meta,
                    "train_shards": len(train_shards),
                    "val_shards": len(val_shards),
                    "total_shards": len(shard_files),
                    "total_size_bytes": total_size_bytes,
                    "total_size_human": _format_bytes(total_size_bytes),
                    "meta_tokens": meta.get("tokens"),
                    "meta_train_tokens": meta.get("train_tokens"),
                    "meta_val_tokens": meta.get("val_tokens"),
                }
            )
            bin_sets.append(record)

    return {
        "tokenizers": tokenizer_files,
        "clean_files": clean_files,
        "manifests": manifest_files,
        "bins": bin_sets,
    }


def _scan_scripts(root: Path) -> list[dict[str, Any]]:
    scripts_root = root / "scripts"
    if not scripts_root.exists():
        return []

    records = []
    for path in sorted(scripts_root.glob("*.py")):
        preview = _preview_text(path, max_chars=3000)
        args = sorted(
            set(re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9_-]+)", preview))
        )
        record = _file_record(path, root)
        record.update(
            {
                "arguments": args,
                "argument_count": len(args),
                "has_argparse": "argparse" in preview,
                "preview": preview,
            }
        )
        records.append(record)
    return records


def _find_run_directories(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []

    directories = []
    for directory in runs_root.rglob("*"):
        if not directory.is_dir():
            continue
        has_artifacts = (
            (directory / "metrics.jsonl").exists()
            or (directory / "errors.jsonl").exists()
            or any(directory.glob("*.pt"))
            or (
                (directory / "samples").exists()
                and any((directory / "samples").glob("*.txt"))
            )
        )
        if has_artifacts:
            directories.append(directory)

    return sorted(directories, key=lambda item: item.stat().st_mtime, reverse=True)


def _summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest_train = next(
        (row for row in reversed(rows) if "train_loss" in row),
        None,
    )
    latest_val = next(
        (row for row in reversed(rows) if "val_loss" in row),
        None,
    )
    val_losses = [
        row["val_loss"]
        for row in rows
        if isinstance(row.get("val_loss"), int | float)
    ]
    latest_step = max((int(row.get("step", 0)) for row in rows), default=0)
    return {
        "latest_step": latest_step,
        "latest_train": latest_train or {},
        "latest_val": latest_val or {},
        "best_val_loss": min(val_losses) if val_losses else None,
    }


def _scan_runs(root: Path) -> list[dict[str, Any]]:
    runs_root = root / "runs"
    records: list[dict[str, Any]] = []

    for directory in _find_run_directories(runs_root):
        relative = directory.relative_to(runs_root)
        parts = relative.parts
        kind = parts[0] if parts else "runs"

        metrics_path = directory / "metrics.jsonl"
        errors_path = directory / "errors.jsonl"
        sample_dir = directory / "samples"

        metrics_rows = _load_jsonl(metrics_path)
        errors_rows = _load_jsonl(errors_path)
        checkpoints = [
            _file_record(path, root) for path in sorted(directory.glob("*.pt"))
        ]
        samples = []
        if sample_dir.exists():
            for sample_path in sorted(sample_dir.glob("*.txt"), reverse=True):
                sample_record = _file_record(sample_path, root)
                sample_record["preview"] = _preview_text(sample_path)
                samples.append(sample_record)

        metric_summary = _summarize_metrics(metrics_rows)
        record = _dir_record(directory, root)
        record.update(
            {
                "kind": kind,
                "metrics_path": (
                    _relative_path(metrics_path, root) if metrics_path.exists() else None
                ),
                "errors_path": (
                    _relative_path(errors_path, root) if errors_path.exists() else None
                ),
                "metrics_rows": metrics_rows,
                "errors_rows": errors_rows,
                "checkpoints": checkpoints,
                "samples": samples,
                "checkpoint_count": len(checkpoints),
                "sample_count": len(samples),
                "error_count": len(errors_rows),
                **metric_summary,
            }
        )
        records.append(record)

    return records


def _build_pipeline(
    *,
    corpora: dict[str, Any],
    data_assets: dict[str, Any],
    runs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    tokenizer_models = [
        asset for asset in data_assets["tokenizers"] if asset["name"].endswith(".model")
    ]
    run_count = len(runs)
    checkpoint_count = sum(run["checkpoint_count"] for run in runs)
    sample_count = sum(run["sample_count"] for run in runs)

    return [
        {
            "name": "Corpus Registry",
            "status": "ready" if corpora["source_count"] else "missing",
            "detail": (
                f"{corpora['source_count']} sources and {corpora['mix_count']} mixes"
                if corpora["source_count"]
                else "Add configs/data/corpora.yaml to describe data sources"
            ),
        },
        {
            "name": "Cleaning",
            "status": "ready" if data_assets["clean_files"] else "missing",
            "detail": (
                f"{len(data_assets['clean_files'])} clean files in data/clean"
                if data_assets["clean_files"]
                else "No clean corpus outputs were found"
            ),
        },
        {
            "name": "Tokenizer",
            "status": "ready" if tokenizer_models else "missing",
            "detail": (
                f"{len(tokenizer_models)} tokenizer model files discovered"
                if tokenizer_models
                else "No SentencePiece .model files were found"
            ),
        },
        {
            "name": "Bin Shards",
            "status": "ready" if data_assets["bins"] else "missing",
            "detail": (
                f"{len(data_assets['bins'])} bin directories with shard metadata"
                if data_assets["bins"]
                else "No bin shard directories were found in data/bins"
            ),
        },
        {
            "name": "Training Runs",
            "status": "ready" if run_count else "missing",
            "detail": (
                f"{run_count} run directories and {checkpoint_count} checkpoints"
                if run_count
                else "No run metrics or checkpoints were found yet"
            ),
        },
        {
            "name": "Samples",
            "status": "ready" if sample_count else "partial" if run_count else "missing",
            "detail": (
                f"{sample_count} generated sample files"
                if sample_count
                else "Runs exist, but no saved text generations were found"
                if run_count
                else "Sample generation will appear once runs produce outputs"
            ),
        },
    ]


def collect_dashboard_state(root: str | Path) -> dict[str, Any]:
    repo_root = _as_path(root)
    configs = _scan_configs(repo_root)
    corpora = _scan_corpora_registry(repo_root)
    data_assets = _scan_data_assets(repo_root)
    scripts = _scan_scripts(repo_root)
    runs = _scan_runs(repo_root)

    module_counts = {}
    for module_name in [
        "cli",
        "data",
        "eval",
        "model",
        "tokenization",
        "train",
        "utils",
    ]:
        module_dir = repo_root / "skull" / module_name
        module_counts[module_name] = (
            len(list(module_dir.glob("*.py"))) if module_dir.exists() else 0
        )

    summary = {
        "repo_root": str(repo_root),
        "config_count": len(configs),
        "train_config_count": sum(1 for item in configs if item["kind"] == "train"),
        "model_config_count": sum(1 for item in configs if item["kind"] == "model"),
        "eval_config_count": sum(1 for item in configs if item["kind"] == "eval"),
        "script_count": len(scripts),
        "test_count": len(list((repo_root / "tests").glob("test_*.py")))
        if (repo_root / "tests").exists()
        else 0,
        "run_count": len(runs),
        "checkpoint_count": sum(run["checkpoint_count"] for run in runs),
        "sample_count": sum(run["sample_count"] for run in runs),
        "bin_directory_count": len(data_assets["bins"]),
        "tokenizer_count": len(
            [
                item
                for item in data_assets["tokenizers"]
                if item["name"].endswith(".model")
            ]
        ),
        "module_counts": module_counts,
    }

    return {
        "summary": summary,
        "pipeline": _build_pipeline(
            corpora=corpora,
            data_assets=data_assets,
            runs=runs,
        ),
        "configs": configs,
        "corpora": corpora,
        "data_assets": data_assets,
        "scripts": scripts,
        "runs": runs,
    }
