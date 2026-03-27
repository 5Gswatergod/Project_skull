# Project Skull

Project Skull 是一個以中文與中英混合語料為主的模組化 LLM 訓練框架。它把資料清洗、tokenizer、bin shard 建置、base pretraining、continued pretraining、SFT、eval 與 sampling 收斂到同一個 repo，並用 YAML config 驅動整個流程。

這個專案目前適合拿來做單機實驗、資料管線驗證，以及小到中型模型的訓練流程整理。整體設計偏向可拆換、可觀測、可恢復，而不是追求極度封裝。

## 設計目標

- `streaming first`：大資料流程優先採流式處理
- `config-driven`：訓練入口都由 YAML 控制
- `multi-bin ready`：天然支援多 source、多 shard
- `resume-safe`：訓練可自動尋找最近 checkpoint 接續
- `modular`：tokenizer、dataset、trainer、model 可以獨立替換

## 目前已實作

- SentencePiece tokenizer wrapper
- decoder-only GPT 模型與 YAML model config
- `BlockBinDataset`、`MultiBinDataset`、`PackedSFTDataset`
- `PretrainTrainer`、`CPTTrainer`、`SFTTrainer`
- CLI：
  - `python -m skull.cli.pretrain`
  - `python -m skull.cli.cpt`
  - `python -m skull.cli.sft`
  - `python -m skull.cli.eval`
  - `python -m skull.cli.sample`
- 常用資料腳本：
  - `scripts/build_clean_corpus.py`
  - `scripts/append_datasets.py`
  - `scripts/train_tokenizer_v4.py`
  - `scripts/build_bins_multishard.py`
  - `scripts/count_tokens.py`
- 基本測試：
  - bin dataset
  - multi-bin dataset
  - SFT dataset
  - model forward
  - CUDA fallback

## 專案狀態

這個 repo 已經可以走完一條完整主線：

1. 準備或清洗純文字
2. 訓練 SentencePiece tokenizer
3. 建立 `.bin` shards
4. 跑 pretraining
5. 跑 CPT 或 SFT
6. 做 eval 與 sample

同時也要注意幾件事：

- `configs/` 裡的 YAML 比較像範例模板，真正開跑前請先核對檔案路徑
- `scripts/train_tokenizer_v4.py` 是目前最完整的 tokenizer 腳本，`v1` 到 `v3` 可視為歷史版本
- 訓練主路徑目前以單機、單進程流程為主，多卡 / distributed 還不是最完整的對外介面
- `device: cuda` 在沒有 CUDA 時會 fallback 到 CPU，但實際訓練會非常慢

## 專案結構

```text
project_skull/
├─ configs/            # data / model / train / eval configs
├─ data/               # clean text、tokenizer、bins、manifests
├─ runs/               # checkpoints、metrics、samples
├─ scripts/            # 資料與訓練輔助腳本
├─ skull/
│  ├─ cli/             # pretrain / cpt / sft / eval / sample 入口
│  ├─ data/            # dataset 與 collator
│  ├─ eval/            # perplexity / generation
│  ├─ model/           # GPT config 與模型元件
│  ├─ tokenization/    # tokenizer wrapper
│  ├─ train/           # trainers / optimizer / scheduler / checkpointing
│  └─ utils/
└─ tests/
```

## 安裝

### 使用虛擬環境

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

### 安裝依賴

如果你想用 editable install：

```bash
pip install -e .[dev]
```

如果你只想快速把依賴裝齊：

```bash
pip install -r requirements.txt
```

### 執行測試

```bash
pytest
```

## 快速開始

如果你已經有：

- tokenizer model
- model config
- train / val `.bin` shards

那可以直接從 pretraining 開始。

### 開跑前先檢查這 4 個欄位

- `tokenizer_model`
- `model_config`
- `train_sources` / `val_sources`
- `run_dir`

特別注意：

- `scripts/train_tokenizer_v4.py` 預設輸出的檔名通常是 `skull_zh_en_128k_bpe.model`
- 如果你的 config 還寫著 `data/tokenizer/skull_zh_en_128k.model`，請改成實際存在的檔名
- shard 清單也要和實際產出的 `train_000.bin`、`train_001.bin` 等對上

### 1. Base pretraining

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

### 2. Eval

```bash
python -m skull.cli.eval \
  --config configs/eval/default_eval.yaml \
  --ckpt runs/pretrain/skull_150m_base/best.pt
```

如果想印成 JSON：

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
  --prompt "你好，請介紹一下台北。" \
  --max_new_tokens 128
```

注意 `sample` CLI 同時需要 `--config` 與 `--ckpt`。

## 從原始文字到訓練

如果你要從原始語料重新建立整條管線，可以照下面的順序。

### 1. 清洗單一文字檔

`scripts/build_clean_corpus.py` 是最簡單的清洗腳本，適合把單一文字檔轉成乾淨純文字：

```bash
python scripts/build_clean_corpus.py \
  --input data/corpus/raw/wiki.txt \
  --output data/clean/wiki.txt
```

它目前主要做：

- 移除 URL
- 移除 HTML tag
- 基本空白整理
- 過短行過濾

### 2. 合併多份 clean text

如果你已經有多份乾淨文字，可以用 `scripts/append_datasets.py` 合併：

```bash
python scripts/append_datasets.py \
  --inputs data/clean/wiki.txt data/clean/books.txt \
  --output data/clean/train_clean.txt \
  --meta data/clean/train.meta.json
```

### 3. 訓練 tokenizer

建議使用 `scripts/train_tokenizer_v4.py`。它支援：

- 多來源抽樣
- ratio / weight 控制
- 基本品質過濾
- exact dedup
- manifest 輸出

範例：

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

如果你只想先產生 sample file、不立刻訓練 SentencePiece：

```bash
python scripts/train_tokenizer_v4.py --skip-train ...
```

### 4. 計算 token 數量

```bash
python scripts/count_tokens.py \
  --input data/clean/train_clean.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model
```

### 5. 建立 multi-shard bin

每個 source 建議分開建 bin，這樣後續比較容易做權重混合。

```bash
python scripts/build_bins_multishard.py \
  --input data/clean/fineweb.txt \
  --tokenizer data/tokenizer/skull_zh_en_128k_bpe.model \
  --out_dir data/bins/fineweb \
  --shard_tokens 50000000 \
  --val_ratio 0.02
```

輸出大致會是：

```text
data/bins/fineweb/
├─ train_000.bin
├─ train_001.bin
├─ ...
├─ val_000.bin
└─ meta.json
```

如果你只需要最簡單的單檔輸出，也可以用 `scripts/build_bins.py`。

### 6. 跑 base pretraining

`configs/train/pretrain_150m.yaml` 的核心欄位通常包含：

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

`train_sources` / `val_sources` 的格式：

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

訓練命令：

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
```

### 7. 跑 continued pretraining

CPT 和 pretraining 的差異主要在：

- 會先載入 `base_ckpt`
- 通常使用更小 learning rate
- source 配比會更偏向目標領域

命令：

```bash
python -m skull.cli.cpt --config configs/train/cpt_150m.yaml
```

### 8. 準備 SFT 資料

SFT JSONL 支援兩種格式。

#### 格式 A：`messages`

```json
{"messages":[
  {"role":"system","content":"你是一個幫助使用者的助理。"},
  {"role":"user","content":"請介紹台北。"},
  {"role":"assistant","content":"台北是台灣的首都..."}
]}
```

#### 格式 B：`prompt` / `response`

```json
{"prompt":"請介紹台北。","response":"台北是台灣的首都..."}
```

`PackedSFTDataset` 目前支援：

- `assistant_only_loss`
- packing
- padding / truncation
- `system` / `tool` / 其他角色標記

當 `assistant_only_loss: true` 時，預設只對 assistant token 計算 loss。

### 9. 跑 SFT

```bash
python -m skull.cli.sft --config configs/train/sft_150m.yaml
```

## 設定檔速查

### train config

- `tokenizer_model`：SentencePiece `.model` 路徑
- `model_config`：模型 YAML，或直接在 train config 內嵌 `model`
- `device`：通常為 `cuda` 或 `cpu`
- `mixed_precision`：`fp16` 或 `bf16`
- `resume`：是否自動從 `run_dir/latest.pt` 接續
- `block_size`：模型上下文長度
- `row_tokens`：通常等於 `block_size + 1`
- `bin_dtype`：目前預設 `uint32`
- `train_nominal_size` / `val_nominal_size`：dataset 名義長度

### model config

`GPTConfig` 目前支援的常用欄位：

- `vocab_size`
- `block_size`
- `n_layer`
- `n_head`
- `n_embd`
- `dropout`
- `bias`
- `norm`：`layernorm` 或 `rmsnorm`
- `pos_encoding`：`absolute` 或 `rope`
- `rope_base`
- `mlp_type`：`gelu` 或 `swiglu`
- `mlp_hidden_mult`
- `tie_word_embeddings`
- `use_checkpointing`

## 資料格式說明

### Pretraining bin

- dtype：`uint32`
- 讀取方式：固定視窗切成 row
- 每 row 長度：`block_size + 1`
- dataset 會輸出：
  - `input_ids = row[:-1]`
  - `labels = row[1:]`

### Multi-source sampling

`MultiBinDataset` 會：

- 先依 source `weight` 抽一個資料源
- 再依 shard 長度比例抽一個 shard
- 最後在 shard 內抽一個 row

這讓你可以用簡單 YAML 做混料，而不需要先把所有來源合成成單一大 bin。

## 輸出內容

以 pretraining 為例，`run_dir` 通常會長這樣：

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

說明：

- `latest.pt`：最近一次保存
- `best.pt`：validation loss 最佳 checkpoint
- `step_*.pt`：階段性快照
- `metrics.jsonl`：train / val metrics
- `samples/`：固定 prompt 的生成結果

如果沒有提供 `val_sources`，就不一定會產生 `best.pt`。

## 常用命令總表

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
python -m skull.cli.cpt --config configs/train/cpt_150m.yaml
python -m skull.cli.sft --config configs/train/sft_150m.yaml
python -m skull.cli.eval --config configs/eval/default_eval.yaml --ckpt runs/pretrain/skull_150m_base/best.pt
python -m skull.cli.sample --config configs/train/pretrain_150m.yaml --ckpt runs/pretrain/skull_150m_base/best.pt --prompt "你好"
```

如果你偏好從 `scripts/` 啟動，也可以使用：

- `scripts/launch_pretrain.py`
- `scripts/launch_cpt.py`
- `scripts/launch_sft.py`

它們本質上只是對應 CLI 的薄包裝。

## 已知注意事項

- 開跑前一定要核對 config 裡的路徑是否真的存在
- `eval` config 必須包含 `eval_sources`
- `sample` CLI 必須有 `--config`
- tokenizer 檔名與 config 不一致是最常見的啟動錯誤之一
- `scripts/export_checkpoint.py` 目前是簡化版工具，使用前請先確認是否符合你的模型結構需求

## 下一步建議

如果你是第一次接手這個專案，最推薦的順序是：

1. 先跑 `pytest`
2. 打開 `configs/train/pretrain_150m.yaml`，確認所有檔案路徑
3. 用一小份資料先走通 tokenizer -> bins -> pretrain
4. 確認 `runs/` 內會正常產生 checkpoint、metrics 與 samples
5. 再把資料量與訓練步數放大
