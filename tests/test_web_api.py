from __future__ import annotations

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from skull.web.jobs import find_job
from skull.web.server import create_app


def _write_text(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _wait_for_status(root, job_id: str, statuses: set[str], timeout: float = 20.0):
    deadline = time.time() + timeout
    latest = None
    while time.time() < deadline:
        latest = find_job(root, job_id)
        if latest and latest.get("status") in statuses:
            return latest
        time.sleep(0.25)
    raise AssertionError(
        f"Timed out waiting for job {job_id} to reach {statuses}. Last state: {latest}"
    )


def _seed_repo(tmp_path):
    _write_text(
        tmp_path / "configs" / "train" / "demo.yaml",
        """
run_name: demo_run
run_dir: runs/pretrain/demo_run
tokenizer_model: data/tokenizer/demo.model
model_config: configs/model/demo.yaml
device: cpu
block_size: 128
max_steps: 50
train_sources:
  - name: wiki
    paths:
      - data/bins/wiki/train_000.bin
""".strip(),
    )
    _write_text(
        tmp_path / "configs" / "model" / "demo.yaml",
        """
vocab_size: 32000
block_size: 128
n_layer: 4
n_head: 4
n_embd: 128
""".strip(),
    )
    _write_text(
        tmp_path / "configs" / "eval" / "demo.yaml",
        """
eval_sources:
  - name: wiki
    type: ppl
""".strip(),
    )
    _write_text(
        tmp_path / "configs" / "data" / "corpora.yaml",
        """
sources:
  - name: wiki
    enabled: true
    format: txt
    domain: knowledge
    lang: zh-en
    clean_output:
      clean_text_path: data/clean/wiki.txt
    bin_output:
      output_dir: data/bins/wiki
mixes:
  base:
    description: demo
    sources:
      - name: wiki
""".strip(),
    )
    _write_text(tmp_path / "data" / "tokenizer" / "demo.model", "model")
    _write_text(tmp_path / "data" / "clean" / "wiki.txt", "hello world\n")

    bins_dir = tmp_path / "data" / "bins" / "wiki"
    bins_dir.mkdir(parents=True, exist_ok=True)
    (bins_dir / "train_000.bin").write_bytes(b"\x00\x00\x00\x01")
    (bins_dir / "meta.json").write_text(json.dumps({"tokens": 10}), encoding="utf-8")

    run_dir = tmp_path / "runs" / "pretrain" / "demo_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / "metrics.jsonl", json.dumps({"step": 5, "train_loss": 4.2}))
    (run_dir / "latest.pt").write_bytes(b"checkpoint")

    _write_text(
        tmp_path / "tests" / "test_demo.py",
        "def test_ok():\n    assert True\n",
    )


def test_dashboard_endpoint_returns_launchpad(tmp_path, monkeypatch):
    _seed_repo(tmp_path)
    monkeypatch.setenv("SKULL_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_root"] == str(tmp_path.resolve())
    assert payload["state"]["summary"]["train_config_count"] == 1
    assert payload["launchpad"]["train_configs"][0]["path"] == "configs/train/demo.yaml"
    assert payload["launchpad"]["checkpoints"] == ["runs/pretrain/demo_run/latest.pt"]


def test_launch_test_job_and_fetch_log_via_api(tmp_path, monkeypatch):
    _seed_repo(tmp_path)
    monkeypatch.setenv("SKULL_REPO_ROOT", str(tmp_path))

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/launch/test",
            json={
                "targets": ["tests/test_demo.py"],
                "label": "api-pytest",
            },
        )
        assert response.status_code == 200
        job = response.json()["job"]

        finished = _wait_for_status(tmp_path, job["id"], {"completed"})

        log_response = client.get(f"/api/jobs/{job['id']}/log")
        assert log_response.status_code == 200
        assert "1 passed" in log_response.json()["log"]

        delete_response = client.request(
            "DELETE",
            f"/api/jobs/{job['id']}",
            json={"delete_log_too": True},
        )

    assert finished["returncode"] == 0
    assert delete_response.status_code == 200
    assert find_job(tmp_path, job["id"]) is None
