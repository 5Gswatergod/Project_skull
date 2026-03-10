# Project Skull

完整 LLM training framework 規格與落地藍圖。

---

## 1. 目標

Project Skull 是一套面向 **中文 / 中英混合語料** 的可擴展 LLM 訓練框架，覆蓋：

- 原始資料匯入與清洗
- tokenizer 訓練
- multi-bin / multi-corpus 資料建置
- pretraining
- continued pretraining
- SFT / instruction tuning
- checkpoint / eval / sampling / resume
- 單機單卡、單機多卡，未來可延伸多機

核心設計原則：

1. **streaming first**：所有大資料流程都優先採流式處理
2. **multi-bin ready**：資料集天然支援 shard / shard mix
3. **config-driven**：所有訓練流程由 yaml 控制
4. **resume-safe**：清洗、build bin、訓練都可恢復
5. **modular**：每個階段都可獨立替換

---

## 2. 專案目錄

```text
project_skull/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ .gitignore
│
├─ configs/
│  ├─ data/
│  │  ├─ corpora.yaml
│  │  └─ tokenizer_zh_en.yaml
│  ├─ model/
│  │  ├─ skull_150m.yaml
│  │  ├─ skull_300m.yaml
│  │  └─ skull_1b.yaml
│  ├─ train/
│  │  ├─ pretrain_150m.yaml
│  │  ├─ cpt_150m.yaml
│  │  └─ sft_150m.yaml
│  └─ eval/
│     └─ default_eval.yaml
│
├─ data/
│  ├─ corpus/
│  │  ├─ raw/
│  │  ├─ staging/
│  │  └─ merged/
│  ├─ clean/
│  │  ├─ train_clean.txt
│  │  ├─ val_clean.txt
│  │  └─ train.meta.json
│  ├─ tokenizer/
│  │  ├─ skull_zh_en_100k.model
│  │  ├─ skull_zh_en_100k.vocab
│  │  └─ tokenizer_sample.txt
│  ├─ bins/
│  │  ├─ wiki/
│  │  │  ├─ train_000.bin
│  │  │  ├─ train_001.bin
│  │  │  └─ meta.json
│  │  ├─ books/
│  │  ├─ web/
│  │  ├─ code/
│  │  └─ mix/
│  └─ manifest/
│     ├─ corpora_manifest.json
│     └─ bin_manifest.json
│
├─ scripts/
│  ├─ append_datasets.py
│  ├─ build_clean_corpus.py
│  ├─ train_tokenizer.py
│  ├─ build_bins.py
│  ├─ build_bins_multishard.py
│  ├─ build_mix_manifest.py
│  ├─ count_tokens.py
│  ├─ launch_pretrain.py
│  ├─ launch_cpt.py
│  ├─ launch_sft.py
│  └─ export_checkpoint.py
│
├─ skull/
│  ├─ __init__.py
│  ├─ tokenization/
│  │  ├─ tokenizer.py
│  │  └─ sentencepiece_wrapper.py
│  ├─ data/
│  │  ├─ cleaning.py
│  │  ├─ manifests.py
│  │  ├─ block_bin_dataset.py
│  │  ├─ multi_bin_dataset.py
│  │  ├─ packed_sft_dataset.py
│  │  └─ collators.py
│  ├─ model/
│  │  ├─ config.py
│  │  ├─ model_gpt.py
│  │  ├─ rope.py
│  │  ├─ norms.py
│  │  ├─ mlp.py
│  │  └─ attention.py
│  ├─ train/
│  │  ├─ trainer_pretrain.py
│  │  ├─ trainer_cpt.py
│  │  ├─ trainer_sft.py
│  │  ├─ checkpointing.py
│  │  ├─ optimizer.py
│  │  ├─ schedulers.py
│  │  └─ losses.py
│  ├─ eval/
│  │  ├─ perplexity.py
│  │  ├─ generation.py
│  │  ├─ benchmark_runner.py
│  │  └─ prompts.py
│  ├─ utils/
│  │  ├─ io.py
│  │  ├─ seed.py
│  │  ├─ logging.py
│  │  ├─ distributed.py
│  │  └─ profiling.py
│  └─ cli/
│     ├─ pretrain.py
│     ├─ cpt.py
│     ├─ sft.py
│     ├─ eval.py
│     └─ sample.py
│
├─ runs/
│  ├─ pretrain/
│  ├─ cpt/
│  └─ sft/
│
└─ tests/
   ├─ test_tokenizer.py
   ├─ test_bin_dataset.py
   ├─ test_multi_bin_dataset.py
   └─ test_model_forward.py
```

---

## 3. 三層資料流

### 3.1 原始語料層

來源可包括：

- txt
- jsonl
- parquet
- 多資料集拼接
- 增量 append

輸出為統一的 `train_clean.txt`。

### 3.2 tokenized / bin 層

將 clean text 經 tokenizer 轉為 token ids，再建成：

- `train_000.bin`
- `train_001.bin`
- `val_000.bin`
- `meta.json`

每個 bin 採 `uint32` token stream 或 row-packed `(block_size + 1)` 格式。

### 3.3 training mix 層

由 manifest 指定多資料集混合權重，例如：

```yaml
sources:
  - name: wiki
    paths:
      - data/bins/wiki/train_000.bin
      - data/bins/wiki/train_001.bin
    weight: 1.0

  - name: books
    paths:
      - data/bins/books/train_000.bin
    weight: 2.0

  - name: code
    paths:
      - data/bins/code/train_000.bin
    weight: 0.5
```

---

## 4. 核心模組

### 4.1 Data cleaning

職責：

- HTML / URL 移除
- Unicode normalization
- 語言與字符過濾
- 垃圾文本過濾
- 長度控制
- dedup（可選）

目前你的 `append_datasets.py` 可作為 `scripts/append_datasets.py` 初始版本。

建議再加：

- MinHash / exact hash dedup
- document-level source tag
- line-level stats
- bad-source blacklist

### 4.2 Tokenizer

採 SentencePiece BPE。

輸入：`train_clean.txt`
輸出：`.model` / `.vocab`

建議保留：

- `unk_id=0`
- `bos_id=1`
- `eos_id=2`
- `pad_id=3`

### 4.3 Multi-shard bin builder

功能：

- streaming build
- flush by token count
- shard rollover
- resume state
- stats / checksum / manifest

每個語料源獨立建 bin，例如：

```text
bins/wiki/train_000.bin
bins/wiki/train_001.bin
bins/books/train_000.bin
```

### 4.4 Dataset loader

包含兩種：

1. `BlockBinDataset`
   - 讀單一 bin
   - 回傳 `(x, y)`

2. `MultiBinDataset`
   - 依權重抽資料源
   - 未來可擴成：
     - source-level weighted sampling
     - within-source shard rotation
     - distributed rank-aware shard split

### 4.5 Model

初始版本為 decoder-only GPT。

建議模型升級順序：

1. 現有 absolute position embedding GPT
2. 改成 RoPE
3. RMSNorm
4. SwiGLU / GeGLU
5. fused attention / Flash Attention
6. bf16 + compile + checkpointing

### 4.6 Trainer

三類 trainer：

- `trainer_pretrain.py`
- `trainer_cpt.py`
- `trainer_sft.py`

共同能力：

- gradient accumulation
- mixed precision
- lr schedule
- eval loop
- checkpoint latest / best
- sample generation
- resume
- metrics logger

---

## 5. 訓練階段設計

### 5.1 Phase A: Base pretraining

目標：從頭訓練 base model。

輸入：

- tokenizer model
- multi-bin corpus
- model config
- pretrain config

輸出：

- `runs/pretrain/skull_150m/latest.pt`
- `best.pt`
- `metrics.jsonl`
- `samples/`

### 5.2 Phase B: Continued pretraining (CPT)

目標：在 base model 上追加領域知識，例如小說、法律、醫療、科技。

輸入：

- base checkpoint
- domain bins
- CPT config

特點：

- 小 learning rate
- 可混通用料避免災難性遺忘

### 5.3 Phase C: SFT

目標：instruction tuning。

資料格式建議：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

或轉為：

```text
<|user|>...
<|assistant|>...
```

SFT dataset 需支援：

- loss masking（只算 assistant token）
- packing
- truncation

### 5.4 Phase D: Eval / Sampling

每個訓練階段都保留：

- perplexity
- held-out loss
- fixed prompts generation
- domain eval set

---

## 6. Project Skull 的最小可用版本（MVP）

### 已有元件

你現在已經有：

- 清洗 append 腳本
- tokenizer wrapper
- GPT model
- pretrain trainer
- 單 bin dataset
- multi-bin dataset 初版
- incremental bin builder

### 還需要補齊

1. `build_bins_multishard.py`
2. `bin manifest` 產生器
3. yaml config 統一格式
4. `trainer_pretrain.py` 改讀 `MultiBinDataset`
5. `trainer_cpt.py`
6. `trainer_sft.py`
7. eval CLI
8. sample CLI
9. token counter
10. tests

---

## 7. 推薦 config 設計

### 7.1 model config

```yaml
name: skull_150m
vocab_size: 100000
block_size: 1024
n_layer: 16
n_head: 12
n_embd: 768
dropout: 0.1
bias: false
norm: layernorm
pos_encoding: absolute
use_checkpointing: true
```

### 7.2 pretrain config

```yaml
run_name: skull_150m_base
run_dir: runs/pretrain/skull_150m_base

tokenizer_model: data/tokenizer/skull_zh_en_100k.model

train_sources:
  - name: wiki
    paths:
      - data/bins/wiki/train_000.bin
      - data/bins/wiki/train_001.bin
    weight: 1.0

  - name: books
    paths:
      - data/bins/books/train_000.bin
    weight: 2.0

val_sources:
  - name: wiki_val
    paths:
      - data/bins/wiki/val_000.bin
    weight: 1.0

batch_size: 8
grad_accum: 8
max_steps: 200000
lr: 3e-4
min_lr: 3e-5
warmup_steps: 2000
weight_decay: 0.1
grad_clip: 1.0
mixed_precision: fp16
log_every: 10
eval_every: 200
sample_every: 500
resume: true
seed: 42
```

### 7.3 SFT config

```yaml
run_name: skull_150m_sft
run_dir: runs/sft/skull_150m_sft

base_ckpt: runs/pretrain/skull_150m_base/best.pt
train_jsonl: data/sft/train.jsonl
val_jsonl: data/sft/val.jsonl

max_seq_len: 2048
batch_size: 4
grad_accum: 8
lr: 1e-5
min_lr: 1e-6
warmup_steps: 200
max_steps: 10000
assistant_only_loss: true
packing: true
resume: true
```

---

## 8. 建議的資料格式

### 8.1 Bin meta.json

```json
{
  "name": "wiki",
  "dtype": "uint32",
  "block_size": 1024,
  "row_tokens": 1025,
  "train_shards": [
    "train_000.bin",
    "train_001.bin"
  ],
  "val_shards": [
    "val_000.bin"
  ],
  "total_train_tokens": 1823456789,
  "total_val_tokens": 4567890,
  "eos_id": 2,
  "tokenizer_model": "data/tokenizer/skull_zh_en_100k.model"
}
```

### 8.2 Corpus manifest

```json
{
  "sources": [
    {
      "name": "wiki",
      "raw_path": "data/corpus/raw/wiki",
      "clean_path": "data/clean/wiki.txt",
      "lang": "zh-en",
      "license": "unknown"
    }
  ]
}
```

---

## 9. 模型升級路線

### v0

- absolute position embedding
- LayerNorm
- GELU MLP
- SentencePiece BPE
- single-node accelerate

### v1

- RoPE
- RMSNorm
- SwiGLU
- cosine LR scheduler
- better eval hooks

### v2

- FSDP / DeepSpeed
- packed pretraining samples
- source temperature sampling
- checkpoint export to HF format

### v3

- instruction template system
- preference optimization（之後可加）
- vLLM / transformers inference export

---

## 10. Project Skull 命名與模組責任

- `skull.data`：資料清洗、manifest、dataset
- `skull.tokenization`：SentencePiece wrapper
- `skull.model`：GPT 本體
- `skull.train`：trainer / checkpoint / optimizer / schedulers
- `skull.eval`：loss、sampling、benchmarks
- `skull.cli`：對外命令列入口

---

## 11. CLI 規劃

```bash
python -m skull.cli.pretrain --config configs/train/pretrain_150m.yaml
python -m skull.cli.cpt --config configs/train/cpt_150m.yaml
python -m skull.cli.sft --config configs/train/sft_150m.yaml
python -m skull.cli.eval --config configs/eval/default_eval.yaml --ckpt runs/pretrain/skull_150m/best.pt
python -m skull.cli.sample --ckpt runs/pretrain/skull_150m/best.pt --prompt "你好，請介紹一下台北。"
```

---

## 12. 你目前最適合的下一步

### Step 1

把現有檔案重新整理進 `project_skull/`：

- `model_gpt.py` → `skull/model/model_gpt.py`
- `tokenizer.py` → `skull/tokenization/tokenizer.py`
- `multi_bin_dataset.py` → `skull/data/multi_bin_dataset.py`
- `utils_io.py` → `skull/data/block_bin_dataset.py`
- `train_pretrain_v3.py` → `skull/train/trainer_pretrain.py`

### Step 2

把 `trainer_pretrain.py` 改成從 yaml 讀 `train_sources` / `val_sources`。

### Step 3

新增 `build_bins_multishard.py`，讓每個 source 輸出：

- `train_000.bin`
- `train_001.bin`
- `val_000.bin`
- `meta.json`

### Step 4

新增 `trainer_cpt.py` 與 `trainer_sft.py`。

---

## 13. 我建議的實作優先順序

1. 整理 package 結構
2. 多 shard bin builder
3. pretrain trainer 接多 source
4. token counter
5. eval / sample CLI
6. CPT trainer
7. SFT trainer
8. RoPE / RMSNorm / SwiGLU

---

## 14. 最終狀態

Project Skull 完成後，你會得到一個可持續擴展的訓練框架：

- 可從零訓練 base LLM
- 可做 continued pretraining
- 可做 instruction tuning
- 可混多個 bin source
- 可自動記錄 metrics / samples / checkpoints
- 可未來升級到更大模型與更多 GPU

---

## 15. 下一個交付建議

下一個實作批次建議直接生成這 6 個檔案：

1. `skull/data/block_bin_dataset.py`
2. `skull/data/multi_bin_dataset.py`
3. `scripts/build_bins_multishard.py`
4. `skull/train/trainer_pretrain.py`
5. `configs/train/pretrain_150m.yaml`
6. `README.md`

這 6 個檔案湊齊後，Project Skull 就會有一個真正可跑的 MVP。

