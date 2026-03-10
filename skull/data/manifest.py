from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def save_json(path: str | Path, data: Any, ensure_ascii: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_corpus_manifest(path: str | Path, sources: list[dict]) -> None:
    save_json(path, {"sources": sources})


def load_corpus_manifest(path: str | Path) -> dict:
    data = load_json(path)
    if "sources" not in data or not isinstance(data["sources"], list):
        raise ValueError(f"Invalid corpus manifest: {path}")
    return data


def save_bin_meta(path: str | Path, meta: dict) -> None:
    save_json(path, meta)


def load_bin_meta(path: str | Path) -> dict:
    data = load_json(path)
    required = [
        "name",
        "dtype",
        "block_size",
        "row_tokens",
        "eos_id",
    ]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in bin meta: {path}")
    return data


def resolve_source_paths(
    paths: Iterable[str | Path],
    base_dir: str | Path | None = None,
    must_exist: bool = True,
) -> list[str]:
    base = Path(base_dir) if base_dir is not None else None
    resolved: list[str] = []

    for p in paths:
        path = Path(p)
        if not path.is_absolute() and base is not None:
            path = base / path
        path = path.resolve()

        if must_exist and not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        resolved.append(str(path))

    return resolved


def expand_manifest_sources(
    manifest: dict,
    base_dir: str | Path | None = None,
    must_exist: bool = True,
) -> list[dict]:
    if "sources" not in manifest:
        raise ValueError("Manifest missing 'sources'")

    out = []
    for src in manifest["sources"]:
        item = dict(src)
        item["paths"] = resolve_source_paths(
            item.get("paths", []),
            base_dir=base_dir,
            must_exist=must_exist,
        )
        out.append(item)
    return out
