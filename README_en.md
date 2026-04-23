# Project Skull

Project Skull is a modular LLM training framework for Chinese and mixed Chinese-English corpora. It provides a config-driven workflow for preparing text, training tokenizers, building binary shards, pretraining GPT-style models, continuing pretraining, supervised fine-tuning, evaluation, sampling, and monitoring runs through a Streamlit web app.

The project is designed for single-machine experiments, data pipeline validation, and small-to-medium model training workflows. It favors readable modules, explicit YAML configs, recoverable runs, and easy inspection over heavy framework abstraction.

## Features

- Config-driven training with YAML files under `configs/`
- Decoder-only GPT model with configurable model size and architecture options
- SentencePiece tokenizer integration
- Streaming-oriented text cleaning and tokenizer preparation scripts
- Single-source and multi-source binary dataset support
- Base pretraining, continued pretraining, and supervised fine-tuning trainers
- Evaluation and sampling CLIs
- Streamlit control panel for launching jobs and monitoring runs
- Pytest coverage for datasets, training utilities, model forward pass, web jobs, and fallback behavior

## Status

Project Skull can run the full local workflow:

1. Prepare or clean plain text corpora.
2. Train or load a SentencePiece tokenizer.
3. Build `.bin` training and validation shards.
4. Run base pretraining.
5. Run continued pretraining or SFT.
6. Evaluate checkpoints and generate samples.
7. Inspect jobs, logs, checkpoints, metrics, and samples in the web app.

The repository is still experiment-oriented. Before launching real training, verify every path in the selected config file.

## Requirements

- Python 3.10+
- PyTorch 2.2+
- NumPy
- PyYAML
- SentencePiece
- Transformers

Optional extras:

- `dev`: pytest
- `accelerate`: Hugging Face Accelerate
- `web`: Streamlit and pandas

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```bash
pip install -e .[dev,web]
```

Install Accelerate support if needed:

```bash
pip install -e .[accelerate]
```

Alternatively, install the pinned basic requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the test suite first:

```bash
pytest
```

Start pretraining from a config:

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

Evaluate a checkpoint:

```bash
python -m skull.cli.eval \
  --config configs/eval/default_eval.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt \
  --print_json
```

Generate a sample:

```bash
python -m skull.cli.sample \
  --config configs/train/pretrain_150m.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt \
  --prompt "Hello, can you introduce Taipei?" \
  --max_new_tokens 128
```

Use Accelerate:

```bash
accelerate launch --num_processes 2 -m skull.cli.pretrain \
  --config configs/train/pretrain_150m.yaml \
  --accelerate
```

## Web App

Project Skull includes a Streamlit app for a simpler local workflow:

- See pipeline readiness at a glance
- Launch train, eval, sample, and test jobs
- Monitor active jobs and logs
- Inspect run metrics, checkpoints, errors, and samples
- Browse configs, data assets, and scripts
- Switch between auto, light, and dark appearance modes

Install the web extra and launch:

```bash
pip install -e .[web]
python -m skull.web
```

After installation, you can also run:

```bash
skull-web
```

## Data Pipeline

### 1. Clean Text

```bash
python scripts/build_clean_corpus.py \
  --input data/corpus/raw/wiki.txt \
  --output data/clean/wiki.txt
```

The cleaner removes URLs, strips basic HTML tags, normalizes whitespace, and filters very short lines.

### 2. Merge Clean Files

```bash
python scripts/append_datasets.py \
  --inputs data/clean/wiki.txt data/clean/books.txt \
  --output data/clean/train_clean.txt \
  --meta data/clean/train.meta.json
```

### 3. Train A Tokenizer

Use `scripts/train_tokenizer_v4.py` for the most complete tokenizer workflow:

```bash
python scripts/train_tokenizer_v4.py \
  --source zh=data/clean/novel.txt \
  --source en=data/clean/fineweb.txt \
  --ratio zh=0.75 \
  --ratio en=0.25 \
  --out-dir data/tokenizer
```

Common outputs:

- `data/tokenizer/<model-prefix>.model`
- `data/tokenizer/<model-prefix>.vocab`
- `data/tokenizer/tokenizer_manifest_128k.json`

### 4. Count Tokens

```bash
python scripts/count_tokens.py \
  --input data/clean/train_clean.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model
```

### 5. Build Binary Shards

```bash
python scripts/build_bins_multishard.py \
  --input data/clean/fineweb.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model \
  --out_dir data/bins/fineweb \
  --shard_tokens 50000000 \
  --val_ratio 0.02
```

Expected output:

```text
data/bins/fineweb/
├─ train_000.bin
├─ train_001.bin
├─ val_000.bin
└─ meta.json
```

For a simple single-file bin, use `scripts/build_bins.py`.

## Training Workflows

### Base Pretraining

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

### Continued Pretraining

```bash
python -m skull.cli.cpt --config configs/train/cpt_150m.yaml
```

CPT loads `base_ckpt`, usually uses a smaller learning rate, and typically emphasizes a target-domain data mix.

### Supervised Fine-Tuning

```bash
python -m skull.cli.sft --config configs/train/sft_150m.yaml
```

SFT JSONL supports either chat-style `messages` records:

```json
{"messages":[{"role":"user","content":"Please introduce Taipei."},{"role":"assistant","content":"Taipei is the capital of Taiwan..."}]}
```

Or simple `prompt` / `response` records:

```json
{"prompt":"Please introduce Taipei.","response":"Taipei is the capital of Taiwan..."}
```

`PackedSFTDataset` supports packing, padding, truncation, role markers, and assistant-only loss.

## Configuration

Check these fields before running any training config:

- `tokenizer_model`
- `model_config`
- `train_sources` / `val_sources`
- `run_dir`

Common train config fields:

- `device`
- `mixed_precision`
- `resume`
- `block_size`
- `row_tokens`
- `bin_dtype`
- `batch_size`
- `grad_accum`
- `max_steps`
- `lr`, `min_lr`, `warmup_steps`
- `log_every`, `eval_every`, `save_every`, `sample_every`

Common model config fields:

- `vocab_size`
- `block_size`
- `n_layer`
- `n_head`
- `n_embd`
- `dropout`
- `norm`
- `pos_encoding`
- `mlp_type`
- `tie_word_embeddings`
- `use_checkpointing`

## Project Layout

```text
Project_skull/
├─ configs/             # data, model, train, and eval configs
├─ data/                # local corpora, tokenizers, bins, and manifests
├─ runs/                # local checkpoints, metrics, errors, and samples
├─ scripts/             # data and training helper scripts
├─ skull/
│  ├─ cli/              # pretrain, cpt, sft, eval, sample entry points
│  ├─ data/             # datasets, manifests, collators
│  ├─ eval/             # evaluation and generation helpers
│  ├─ model/            # GPT model components
│  ├─ tokenization/     # tokenizer wrappers
│  ├─ train/            # trainers, optimizer, scheduler, checkpointing
│  ├─ utils/            # shared utilities
│  └─ web/              # Streamlit app and web job runner
├─ tests/               # pytest suite
├─ pyproject.toml
└─ requirements.txt
```

`data/`, `runs/`, `.skull_web/`, caches, and build metadata are local artifacts and should not be treated as source code.

## Development

Run all tests:

```bash
pytest
```

Run only web-related tests:

```bash
pytest tests/test_web_dashboard_data.py tests/test_web_jobs.py
```

Useful launch wrappers:

- `scripts/launch_pretrain.py`
- `scripts/launch_cpt.py`
- `scripts/launch_sft.py`

These are thin wrappers around the module CLIs.

## Troubleshooting

- Config paths are templates. Verify they point to files that exist locally.
- Tokenizer filename mismatches are a common startup error.
- `sample` requires both `--config` and `--ckpt`.
- `eval` configs must include `eval_sources`.
- `device: cuda` falls back to CPU if CUDA is unavailable, but training will be slow.
- If no validation sources are configured, `best.pt` may not be produced.

## Recommended First Run

1. Run `pytest`.
2. Open `configs/train/pretrain_150m.yaml`.
3. Verify `tokenizer_model`, `model_config`, data shard paths, and `run_dir`.
4. Use a small corpus to run tokenizer -> bins -> pretraining.
5. Confirm that `runs/` contains checkpoints, metrics, and samples.
6. Increase data size and training steps after the small run is healthy.
