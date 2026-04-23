from __future__ import annotations

import json

from skull.web.data import collect_dashboard_state


def _write_text(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_collect_dashboard_state(tmp_path):
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
val_sources:
  - name: wiki_val
    paths:
      - data/bins/wiki/val_000.bin
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
        tmp_path / "configs" / "data" / "corpora.yaml",
        """
sources:
  - name: wiki
    enabled: true
    format: txt
    domain: knowledge
    lang: zh-en
    tokenizer_sampling_weight: 1.5
    pretrain_weight: 1.0
    cpt_weight: 0.5
    clean_output:
      clean_text_path: data/clean/wiki.txt
    bin_output:
      output_dir: data/bins/wiki
mixes:
  pretrain_base:
    description: Demo mix
    sources:
      - name: wiki
        weight: 1.0
    val_sources:
      - name: wiki_val
        weight: 1.0
""".strip(),
    )

    _write_text(tmp_path / "scripts" / "launch_demo.py", "print('demo')\n")
    _write_text(
        tmp_path / "tests" / "test_demo.py",
        "def test_ok():\n    assert True\n",
    )
    _write_text(
        tmp_path / "skull" / "train" / "trainer_demo.py",
        "class Demo:\n    pass\n",
    )
    _write_text(
        tmp_path / "skull" / "model" / "model_demo.py",
        "class Model:\n    pass\n",
    )
    _write_text(tmp_path / "data" / "tokenizer" / "demo.model", "model")
    _write_text(tmp_path / "data" / "clean" / "wiki.txt", "hello world\n")
    _write_text(tmp_path / "data" / "manifest" / "stats.json", "{}")

    bins_dir = tmp_path / "data" / "bins" / "wiki"
    bins_dir.mkdir(parents=True, exist_ok=True)
    (bins_dir / "train_000.bin").write_bytes(b"\x00\x00\x00\x01")
    (bins_dir / "val_000.bin").write_bytes(b"\x00\x00\x00\x02")
    (bins_dir / "meta.json").write_text(
        json.dumps(
            {
                "tokens": 200,
                "train_tokens": 180,
                "val_tokens": 20,
            }
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "pretrain" / "demo_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        run_dir / "metrics.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 10,
                        "train_loss": 4.2,
                        "train_acc": 0.21,
                        "lr": 1e-4,
                    }
                ),
                json.dumps({"step": 20, "val_loss": 3.7, "val_acc": 0.28}),
                json.dumps(
                    {
                        "step": 30,
                        "train_loss": 3.9,
                        "train_acc": 0.26,
                        "lr": 9e-5,
                    }
                ),
            ]
        ),
    )
    _write_text(
        run_dir / "errors.jsonl",
        json.dumps(
            {
                "step": 25,
                "stage": "train_step",
                "action": "continue",
                "error_type": "RuntimeError",
                "message": "out of memory",
            }
        ),
    )
    (run_dir / "latest.pt").write_bytes(b"checkpoint")
    _write_text(run_dir / "samples" / "step_00000030_0.txt", "sample output")

    state = collect_dashboard_state(tmp_path)

    assert state["summary"]["train_config_count"] == 1
    assert state["summary"]["model_config_count"] == 1
    assert state["summary"]["script_count"] == 1
    assert state["summary"]["test_count"] == 1
    assert state["summary"]["checkpoint_count"] == 1
    assert state["summary"]["tokenizer_count"] == 1

    assert state["corpora"]["source_count"] == 1
    assert state["corpora"]["mix_count"] == 1
    assert state["data_assets"]["bins"][0]["meta_tokens"] == 200
    assert state["pipeline"][0]["status"] == "ready"
    assert state["scripts"][0]["relative_path"] == "scripts/launch_demo.py"

    run = state["runs"][0]
    assert run["name"] == "demo_run"
    assert run["latest_step"] == 30
    assert run["best_val_loss"] == 3.7
    assert run["checkpoint_count"] == 1
    assert run["sample_count"] == 1
    assert run["error_count"] == 1
