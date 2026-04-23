# Project Skull

Project Skull 是一個面向中文與中英混合語料的模組化 LLM 訓練框架。它把文字清洗、tokenizer 訓練、binary shard 建置、base pretraining、continued pretraining、SFT、evaluation、sampling，以及 Streamlit Web 監控介面整理在同一個 repo 裡，並用 YAML config 驅動主要流程。

這個專案適合單機實驗、資料管線驗證，以及小到中型模型訓練流程整理。設計上偏向模組清楚、設定明確、可恢復、可觀測，而不是把所有細節藏在高度封裝後面。

## 功能特色

- 使用 `configs/` 內的 YAML 檔驅動訓練流程
- 支援 decoder-only GPT 模型與可調整的模型設定
- 整合 SentencePiece tokenizer
- 提供偏 streaming 的文字清洗與 tokenizer 準備腳本
- 支援單來源與多來源 binary dataset
- 支援 base pretraining、continued pretraining、supervised fine-tuning
- 提供 evaluation 與 sampling CLI
- 提供 Streamlit control panel，可啟動 job、監控 log 與檢視 run artifacts
- 使用 pytest 覆蓋 dataset、training utilities、model forward、web jobs 與 fallback 行為

## 專案狀態

Project Skull 目前可以走完完整本地流程：

1. 準備或清洗純文字語料。
2. 訓練或載入 SentencePiece tokenizer。
3. 建立 `.bin` training / validation shards。
4. 執行 base pretraining。
5. 執行 continued pretraining 或 SFT。
6. 對 checkpoint 做 evaluation 與 sample generation。
7. 在 Web App 中檢視 jobs、logs、checkpoints、metrics 與 samples。

這仍然是一個偏實驗導向的 repo。真正開跑訓練前，請先檢查所選 config 裡的每一個路徑。

## 系統需求

- Python 3.10+
- PyTorch 2.2+
- NumPy
- PyYAML
- SentencePiece
- Transformers

可選 extras：

- `dev`：pytest
- `accelerate`：Hugging Face Accelerate
- `web`：Streamlit 與 pandas

## 安裝

建立並啟動 virtual environment：

```bash
python -m venv .venv
```

macOS / Linux：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

用 editable mode 安裝：

```bash
pip install -e .[dev,web]
```

如果需要 Accelerate：

```bash
pip install -e .[accelerate]
```

或只安裝基本 requirements：

```bash
pip install -r requirements.txt
```

## 快速開始

先跑測試：

```bash
pytest
```

從 config 開始 pretraining：

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

評估 checkpoint：

```bash
python -m skull.cli.eval \
  --config configs/eval/default_eval.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt \
  --print_json
```

產生 sample：

```bash
python -m skull.cli.sample \
  --config configs/train/pretrain_150m.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt \
  --prompt "你好，請介紹一下台北。" \
  --max_new_tokens 128
```

使用 Accelerate：

```bash
accelerate launch --num_processes 2 -m skull.cli.pretrain \
  --config configs/train/pretrain_150m.yaml \
  --accelerate
```

## Web App

Project Skull 內建 Streamlit app，提供更簡單的本地操作流程：

- 快速檢查 pipeline readiness
- 啟動 train、eval、sample 與 test jobs
- 監控 active jobs 與 logs
- 檢視 run metrics、checkpoints、errors 與 samples
- 瀏覽 configs、data assets 與 scripts
- 支援 auto、light、dark appearance modes

安裝 web extra 並啟動：

```bash
pip install -e .[web]
python -m skull.web
```

安裝完成後也可以使用 console script：

```bash
skull-web
```

## 資料管線

### 1. 清洗文字

```bash
python scripts/build_clean_corpus.py \
  --input data/corpus/raw/wiki.txt \
  --output data/clean/wiki.txt
```

清洗腳本會移除 URL、簡單 HTML tags、整理空白，並過濾過短行。

### 2. 合併 clean files

```bash
python scripts/append_datasets.py \
  --inputs data/clean/wiki.txt data/clean/books.txt \
  --output data/clean/train_clean.txt \
  --meta data/clean/train.meta.json
```

### 3. 訓練 tokenizer

建議使用目前最完整的 `scripts/train_tokenizer_v4.py`：

```bash
python scripts/train_tokenizer_v4.py \
  --source zh=data/clean/novel.txt \
  --source en=data/clean/fineweb.txt \
  --ratio zh=0.75 \
  --ratio en=0.25 \
  --out-dir data/tokenizer
```

常見輸出：

- `data/tokenizer/<model-prefix>.model`
- `data/tokenizer/<model-prefix>.vocab`
- `data/tokenizer/tokenizer_manifest_128k.json`

### 4. 計算 token 數量

```bash
python scripts/count_tokens.py \
  --input data/clean/train_clean.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model
```

### 5. 建立 binary shards

```bash
python scripts/build_bins_multishard.py \
  --input data/clean/fineweb.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model \
  --out_dir data/bins/fineweb \
  --shard_tokens 50000000 \
  --val_ratio 0.02
```

預期輸出：

```text
data/bins/fineweb/
├─ train_000.bin
├─ train_001.bin
├─ val_000.bin
└─ meta.json
```

如果只需要最簡單的單檔 bin，可以使用 `scripts/build_bins.py`。

## 訓練流程

### Base Pretraining

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

### Continued Pretraining

```bash
python -m skull.cli.cpt --config configs/train/cpt_150m.yaml
```

CPT 會載入 `base_ckpt`，通常使用較小 learning rate，資料混合也會更偏向目標領域。

### Supervised Fine-Tuning

```bash
python -m skull.cli.sft --config configs/train/sft_150m.yaml
```

SFT JSONL 支援 chat-style `messages`：

```json
{"messages":[{"role":"user","content":"請介紹台北。"},{"role":"assistant","content":"台北是台灣的首都..."}]}
```

也支援簡單的 `prompt` / `response`：

```json
{"prompt":"請介紹台北。","response":"台北是台灣的首都..."}
```

`PackedSFTDataset` 支援 packing、padding、truncation、role markers，以及 assistant-only loss。

## 設定檔

訓練前請先檢查這些欄位：

- `tokenizer_model`
- `model_config`
- `train_sources` / `val_sources`
- `run_dir`

常見 train config 欄位：

- `device`
- `mixed_precision`
- `resume`
- `block_size`
- `row_tokens`
- `bin_dtype`
- `batch_size`
- `grad_accum`
- `max_steps`
- `lr`、`min_lr`、`warmup_steps`
- `log_every`、`eval_every`、`save_every`、`sample_every`

常見 model config 欄位：

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

## 專案結構

```text
Project_skull/
├─ configs/             # data、model、train、eval configs
├─ data/                # local corpora、tokenizers、bins、manifests
├─ runs/                # local checkpoints、metrics、errors、samples
├─ scripts/             # data 與 training helper scripts
├─ skull/
│  ├─ cli/              # pretrain、cpt、sft、eval、sample entry points
│  ├─ data/             # datasets、manifests、collators
│  ├─ eval/             # evaluation 與 generation helpers
│  ├─ model/            # GPT model components
│  ├─ tokenization/     # tokenizer wrappers
│  ├─ train/            # trainers、optimizer、scheduler、checkpointing
│  ├─ utils/            # shared utilities
│  └─ web/              # Streamlit app 與 web job runner
├─ tests/               # pytest suite
├─ pyproject.toml
└─ requirements.txt
```

`data/`、`runs/`、`.skull_web/`、cache 與 build metadata 都屬於 local artifacts，不應視為 source code。

## 開發

跑完整測試：

```bash
pytest
```

只跑 web 相關測試：

```bash
pytest tests/test_web_dashboard_data.py tests/test_web_jobs.py
```

常用 launch wrappers：

- `scripts/launch_pretrain.py`
- `scripts/launch_cpt.py`
- `scripts/launch_sft.py`

它們本質上是 module CLIs 的薄包裝。

## Troubleshooting

- `configs/` 裡的路徑多半是範例模板，開跑前請確認檔案真的存在。
- tokenizer 檔名與 config 不一致是最常見的啟動錯誤之一。
- `sample` 必須同時提供 `--config` 與 `--ckpt`。
- `eval` config 必須包含 `eval_sources`。
- `device: cuda` 在沒有 CUDA 時會 fallback 到 CPU，但實際訓練會很慢。
- 如果沒有設定 `val_sources`，不一定會產生 `best.pt`。

## 建議第一次執行順序

1. 跑 `pytest`。
2. 打開 `configs/train/pretrain_150m.yaml`。
3. 檢查 `tokenizer_model`、`model_config`、data shard paths 與 `run_dir`。
4. 用小語料先走 tokenizer -> bins -> pretraining。
5. 確認 `runs/` 內有正常產生 checkpoints、metrics 與 samples。
6. 小流程健康後，再放大資料量與訓練步數。
