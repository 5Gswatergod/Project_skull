# Project Skull

Project Skull is a modular LLM training framework focused on Chinese and mixed Chinese-English corpora. It brings data cleaning, tokenizer training, bin shard building, base pretraining, continued pretraining, SFT, evaluation, and sampling into a single repo, with the entire workflow driven by YAML configs.

This project is currently well suited for single-machine experiments, data pipeline validation, and organizing training workflows for small to medium-sized models. The overall design emphasizes replaceability, observability, and resumability rather than heavy abstraction.

## Design Goals

- `streaming first`: prefer streaming for large-scale data processing
- `config-driven`: all training entry points are controlled by YAML
- `multi-bin ready`: supports multiple sources and multiple shards out of the box
- `resume-safe`: training can automatically find the latest checkpoint and resume
- `modular`: tokenizer, dataset, trainer, and model can be swapped independently

## Currently Implemented

- SentencePiece tokenizer wrapper
- Decoder-only GPT model and YAML model config
- `BlockBinDataset`, `MultiBinDataset`, `PackedSFTDataset`
- `PretrainTrainer`, `CPTTrainer`, `SFTTrainer`
- CLIs:
  - `python -m skull.cli.pretrain`
  - `python -m skull.cli.cpt`
  - `python -m skull.cli.sft`
  - `python -m skull.cli.eval`
  - `python -m skull.cli.sample`
- Common data scripts:
  - `scripts/build_clean_corpus.py`
  - `scripts/append_datasets.py`
  - `scripts/train_tokenizer_v4.py`
  - `scripts/build_bins_multishard.py`
  - `scripts/count_tokens.py`
- Basic tests:
  - bin dataset
  - multi-bin dataset
  - SFT dataset
  - model forward
  - CUDA fallback

## Project Status

This repo can already run through a complete end-to-end workflow:

1. Prepare or clean plain text
2. Train a SentencePiece tokenizer
3. Build `.bin` shards
4. Run pretraining
5. Run CPT or SFT
6. Run evaluation and sampling

There are also a few things to keep in mind:

- The YAML files in `configs/` are closer to example templates, so verify all file paths before running anything for real.
- `scripts/train_tokenizer_v4.py` is currently the most complete tokenizer script. `v1` through `v3` can be treated as historical versions.
- The main training path is currently centered on single-machine, single-process workflows. Multi-GPU / distributed support is not yet a polished public interface.
- `device: cuda` falls back to CPU when CUDA is unavailable, but actual training will be very slow.

## Project Structure

```text
project_skull/
├─ configs/            # data / model / train / eval configs
├─ data/               # clean text, tokenizer, bins, manifests
├─ runs/               # checkpoints, metrics, samples
├─ scripts/            # data and training helper scripts
├─ skull/
│  ├─ cli/             # pretrain / cpt / sft / eval / sample entry points
│  ├─ data/            # datasets and collators
│  ├─ eval/            # perplexity / generation
│  ├─ model/           # GPT config and model components
│  ├─ tokenization/    # tokenizer wrapper
│  ├─ train/           # trainers / optimizer / scheduler / checkpointing
│  └─ utils/
└─ tests/
```

## Installation

### Use a Virtual Environment

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

### Install Dependencies

If you want an editable install:

```bash
pip install -e .[dev]
```

If you just want to install the dependencies quickly:

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
pytest
```

## Quick Start

If you already have:

- a tokenizer model
- a model config
- train / val `.bin` shards

then you can start directly from pretraining.

### Check These 4 Fields Before Running

- `tokenizer_model`
- `model_config`
- `train_sources` / `val_sources`
- `run_dir`

Important notes:

- `scripts/train_tokenizer_v4.py` usually outputs a filename like `skull_zh_en_128k_bpe.model`
- If your config still points to `data/tokenizer/skull_zh_en_128k.model`, change it to the filename that actually exists
- Your shard list also needs to match the real outputs such as `train_000.bin`, `train_001.bin`, and so on

### 1. Base Pretraining

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

### 2. Eval

```bash
python -m skull.cli.eval \
  --config configs/eval/default_eval.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt
```

If you want JSON output:

```bash
python -m skull.cli.eval \
  --config configs/eval/default_eval.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt \
  --print_json
```

### 3. Sample

```bash
python -m skull.cli.sample \
  --config configs/train/pretrain_150m.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt \
  --prompt "Hello, can you introduce Taipei?" \
  --max_new_tokens 128
```

Note that the `sample` CLI requires both `--config` and `--ckpt`.

## From Raw Text to Training

If you want to rebuild the entire pipeline from raw corpora, follow the order below.

### 1. Clean a Single Text File

`scripts/build_clean_corpus.py` is the simplest cleaning script and is suitable for turning a single text file into clean plain text:

```bash
python scripts/build_clean_corpus.py \
  --input data/corpus/raw/wiki.txt \
  --output data/clean/wiki.txt
```

It currently mainly does the following:

- remove URLs
- remove HTML tags
- basic whitespace cleanup
- filter out overly short lines

### 2. Merge Multiple Clean Text Files

If you already have multiple clean text files, you can merge them with `scripts/append_datasets.py`:

```bash
python scripts/append_datasets.py \
  --inputs data/clean/wiki.txt data/clean/books.txt \
  --output data/clean/train_clean.txt \
  --meta data/clean/train.meta.json
```

### 3. Train the Tokenizer

It is recommended to use `scripts/train_tokenizer_v4.py`. It supports:

- multi-source sampling
- ratio / weight control
- basic quality filtering
- exact dedup
- manifest output

Example:

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

If you only want to generate the sample file first and not train SentencePiece immediately:

```bash
python scripts/train_tokenizer_v4.py --skip-train ...
```

### 4. Count Tokens

```bash
python scripts/count_tokens.py \
  --input data/clean/train_clean.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model
```

### 5. Build Multi-Shard Bins

It is recommended to build bins separately for each source, which makes later weighted mixing much easier.

```bash
python scripts/build_bins_multishard.py \
  --input data/clean/fineweb.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model \
  --out_dir data/bins/fineweb \
  --shard_tokens 50000000 \
  --val_ratio 0.02
```

The output will look roughly like this:

```text
data/bins/fineweb/
├─ train_000.bin
├─ train_001.bin
├─ ...
├─ val_000.bin
└─ meta.json
```

If you only need the simplest single-file output, you can also use `scripts/build_bins.py`.

### 6. Run Base Pretraining

The core fields in `configs/train/pretrain_150m.yaml` usually include:

- `run_name`
- `run_dir`
- `tokenizer_model`
- `model_config`
- `device`
- `mixed_precision`
- `block_size`
- `bin_dtype`
- `row_tokens`
- `train_sources`
- `val_sources`
- `batch_size`
- `grad_accum`
- `max_steps`
- `lr` / `min_lr` / `warmup_steps`
- `log_every` / `eval_every` / `save_every` / `sample_every`

The format of `train_sources` / `val_sources`:

```yaml
train_sources:
  - name: fineweb
    paths:
      - data/bins/fineweb/train_000.bin
      - data/bins/fineweb/train_001.bin
    weight: 1.0

  - name: novel
    paths:
      - data/bins/novel/train_000.bin
      - data/bins/novel/train_001.bin
    weight: 2.0
```

Training command:

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

### 7. Run Continued Pretraining

The main differences between CPT and base pretraining are:

- it loads `base_ckpt` first
- it usually uses a smaller learning rate
- source mixing is more biased toward the target domain

Command:

```bash
python -m skull.cli.cpt --config configs/train/cpt_150m.yaml
```

### 8. Prepare SFT Data

SFT JSONL supports two formats.

#### Format A: `messages`

```json
{"messages":[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"Please introduce Taipei."},
  {"role":"assistant","content":"Taipei is the capital of Taiwan..."}
]}
```

#### Format B: `prompt` / `response`

```json
{"prompt":"Please introduce Taipei.","response":"Taipei is the capital of Taiwan..."}
```

`PackedSFTDataset` currently supports:

- `assistant_only_loss`
- packing
- padding / truncation
- `system` / `tool` / other role markers

When `assistant_only_loss: true`, only assistant tokens contribute to the loss by default.

### 9. Run SFT

```bash
python -m skull.cli.sft --config configs/train/sft_150m.yaml
```

## Config Cheat Sheet

### Train Config

- `tokenizer_model`: path to the SentencePiece `.model`
- `model_config`: model YAML, or embed `model` directly inside the train config
- `device`: usually `cuda` or `cpu`
- `mixed_precision`: `fp16` or `bf16`
- `resume`: whether to automatically resume from `run_dir/latest.pt`
- `block_size`: model context length
- `row_tokens`: usually equal to `block_size + 1`
- `bin_dtype`: currently defaults to `uint32`
- `train_nominal_size` / `val_nominal_size`: nominal dataset lengths

### Model Config

Common fields currently supported by `GPTConfig`:

- `vocab_size`
- `block_size`
- `n_layer`
- `n_head`
- `n_embd`
- `dropout`
- `bias`
- `norm`: `layernorm` or `rmsnorm`
- `pos_encoding`: `absolute` or `rope`
- `rope_base`
- `mlp_type`: `gelu` or `swiglu`
- `mlp_hidden_mult`
- `tie_word_embeddings`
- `use_checkpointing`

## Data Format Notes

### Pretraining Bin

- dtype: `uint32`
- reading pattern: fixed windows sliced into rows
- row length: `block_size + 1`
- the dataset outputs:
  - `input_ids = row[:-1]`
  - `labels = row[1:]`

### Multi-Source Sampling

`MultiBinDataset` will:

- sample a source according to source `weight`
- sample a shard according to shard length ratio
- sample a row within that shard

This lets you mix data sources with a simple YAML config, without first merging everything into one giant bin.

## Outputs

For pretraining, `run_dir` usually looks like this:

```text
runs/pretrain/<run_name>/
├─ latest.pt
├─ best.pt
├─ step_00000200.pt
├─ step_00000400.pt
├─ metrics.jsonl
└─ samples/
   ├─ step_00000500_0.txt
   └─ step_00000500_1.txt
```

Notes:

- `latest.pt`: the most recent save
- `best.pt`: the checkpoint with the best validation loss
- `step_*.pt`: periodic snapshots
- `metrics.jsonl`: train / val metrics
- `samples/`: generations from fixed prompts

If `val_sources` are not provided, `best.pt` may not be produced.

## Common Commands

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
python -m skull.cli.cpt --config configs/train/cpt_150m.yaml
python -m skull.cli.sft --config configs/train/sft_150m.yaml
python -m skull.cli.eval --config configs/eval/default_eval.yaml --ckpt runs/pretrain/skull_150m_base/best.pt
python -m skull.cli.sample --config configs/train/pretrain_150m.yaml --ckpt runs/pretrain/skull_150m_base/best.pt --prompt "Hello"
```

If you prefer launching from `scripts/`, you can also use:

- `scripts/launch_pretrain.py`
- `scripts/launch_cpt.py`
- `scripts/launch_sft.py`

These are essentially thin wrappers around the corresponding CLIs.

## Known Notes

- Always verify that the paths in the config actually exist before running
- The `eval` config must include `eval_sources`
- The `sample` CLI must include `--config`
- A mismatch between the tokenizer filename and the config is one of the most common startup errors
- `scripts/export_checkpoint.py` is currently a simplified utility, so confirm it matches your model architecture before using it

## Recommended Next Steps

If this is your first time picking up this project, the recommended order is:

1. Run `pytest`
2. Open `configs/train/pretrain_150m.yaml` and verify all file paths
3. Use a small dataset to run through tokenizer -> bins -> pretrain
4. Confirm that `runs/` produces checkpoints, metrics, and samples correctly
5. Then scale up the data size and training steps
