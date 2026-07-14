# VSeek-R1

**Official codebase for VSeek-R1** — a tool-using video agent that learns to *seek* answer-critical evidence in long videos via reinforcement learning (GRPO), rather than ingesting dense frame stacks end-to-end.

> **Slurm / HPC (TACC Vista):** see [`README_SLURM.md`](README_SLURM.md).

<!-- Update these badges once the camera-ready / arXiv links are final
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](PAPER_URL)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
-->

> **Path note (local vs Slurm):** defaults on this branch (`v0.6.1`) point at the **local lab** layout under `/nas/mars/...` and `/home/.../vseek/dataset`. The `harsh_tacc` / Slurm configs use **TACC `$SCRATCH` / `$WORK`** paths instead. Always rewrite dataset, index, ViCLIP, and parquet paths for your machine before launching the retriever or training — see [Path conventions](#path-conventions-local-vs-tacc).

---

## Table of Contents

- [Overview](#overview)
- [Path conventions (local vs TACC)](#path-conventions-local-vs-tacc)
- [Requirements](#requirements)
- [Installation](#installation)
- [Download Datasets](#download-datasets)
- [Process Datasets](#process-datasets)
- [Run the Retriever Server](#run-the-retriever-server)
- [Inference](#inference)
- [Training (GRPO)](#training-grpo)
- [Evaluation](#evaluation)
- [Repository Layout](#repository-layout)
- [Citation](#citation)

---

## Path conventions (local vs TACC)

Configs and launch scripts hard-code filesystem roots. **`v0.6.1` assumes a local shared NAS; `harsh_tacc` retargets only some of those files to Vista.** Mixing branches or launching training scripts without checking paths will break the retrieval server and tool calls.

### What actually differs between branches

Audited with `git grep` over both trees (excluding `vendor/`). These are the files whose **dataset / index / weight roots change** `nas`/`home` ↔ `scratch`/`work`:

| File | Local (`v0.6.1`) | TACC (`harsh_tacc`) |
|------|------------------|---------------------|
| `src/vseek/config/retriever/config.yaml` | `/nas/mars/dataset/...`, `/home/.../vseek/dataset`, ViCLIP under `/nas/mars/model_weights/`, port **9005** | `$SCRATCH/.../datasets/...`, `$SCRATCH/.../vseek`, ViCLIP under `$SCRATCH/.../model_weights/`, port **9000** |
| `src/vseek/config/retriever/toolconfig.yaml` | same local roots (note: MLVU is `/nas/mars/dataset/MLVU` here vs `.../MLVU/MLVU` in `config.yaml`) | `$SCRATCH` roots; MLVU uses `.../MLVU/MLVU` |
| `scripts/tacc/*` (harsh only) | n/a | parquet / checkpoints under `$WORK/.../vista/...`; `PROJECT_ROOT` fallback `/home1/.../VSeek-R1`; `HF_HOME` → `$WORK` or `$SCRATCH` |

| Resource | Local (`v0.6.1`) | TACC (`harsh_tacc` / Slurm) |
|----------|------------------|-----------------------------|
| Raw / burned videos | `/nas/mars/dataset/...` | `$SCRATCH/.../datasets/...` |
| FAISS / frame indexes | `/home/<user>/vseek/dataset` | `$SCRATCH/.../vseek` |
| ViCLIP weights | `/nas/mars/model_weights/viclip/` | `$SCRATCH/.../model_weights/viclip/` |
| RL parquet (Slurm jobs) | `/nas/mars/vseek/...` or per-dataset `window_8/` | `$WORK/.../vista/vseek/tagsummary/{train,test}.parquet` |
| Checkpoints (Slurm jobs) | local `checkpoints/` | `$WORK/.../vista/checkpoints` or `$SCRATCH/checkpoints` |
| Retriever port | `9005` | `9000` |

### Still local (`/nas/mars`) on *both* branches — override on TACC

These were **not** rewritten on `harsh_tacc`. On slurm you must override them (CLI / env / edit) or avoid them in favor of `scripts/tacc/`:

- `scripts/train/*.sh` — `data.train_files` / `data.val_files` still under `/nas/mars/...`
- `scripts/data_ops/preprocess_*.sh`, `scripts/utils/unzip_videos.sh` — default `DATASET_PATH` / `INDEX_PATH` / `CHUNKS_DIR` under `/nas/mars` (and CGBench preprocess defaults `INDEX_PATH` to `/home/hg22723/vseek/dataset`)
- `scripts/evals/vllm_vseek_tests_ckpt.sh` — per-dataset parquet roots under `/nas/mars/dataset/.../window_8/`
- Scratch / demo scripts with hardcoded `PROJECT_ROOT=/home/hg22723/projects/VSeek-R1` (e.g. `scripts/run_agent_data_vllmrollout.py`, `scripts/run_api_eval.py`, `scripts/run_video_tree.py`, `scripts/sftdata/run_gpt_agent_data.py`)
- `scripts/puls/*.sh`, VideoTree baseline wrappers, and some notebooks — local-only paths (many exist only on `v0.6.1`)

### TACC/SLURM only paths `harsh_tacc`

- `additional_readme.md` mentions indexes/data under **`$WORK/.../vista/{vseek,datasets}`**, while `config.yaml` / `toolconfig.yaml` on that branch point at **`$SCRATCH/.../{vseek,datasets}`**. Confirm which tree you actually populated.
- `run_vseek_job*.slurm` hard-codes `CHECKPOINT_DIR` / parquet under `/work/11123/harshgoel99/...`; `run_vseek_all_jobs*.slurm` often uses `${SCRATCH}/checkpoints` and env-provided `TRAIN_DATA_PATH`.
- `HF_HOME` is `$WORK/huggingface` in some jobs and `$SCRATCH/huggingface` in others / `ray_wrapper.sh`.
- Slurm `PROJECT_ROOT` fallback is `/home1/11123/harshgoel99/VSeek-R1` (TACC home), not the lab `/home/hg22723/projects/...`.

Files to touch (or override via Hydra) when moving between local and slurm:

- `src/vseek/config/retriever/config.yaml`
- `src/vseek/config/retriever/toolconfig.yaml`
- `scripts/tacc/run_vseek_job.slurm` / `vseek_grpo_tag_qvl34Bt_vllm_tagsummary.sh`
- Any `scripts/train/*.sh` or preprocess/eval script you still invoke on that machine
- Optionally `src/vseek/setting.py`

On TACC, prefer Hydra / CLI overrides over committing personal `$SCRATCH` / `$WORK` paths. Full Slurm examples: [`README_SLURM.md`](README_SLURM.md#path-conventions-tacc-vs-local).

---

## Overview

VSeek-R1 couples a vision–language model with a **video retrieval toolkit** (ViCLIP + FAISS indexes over temporal windows). The agent interacts in multi-turn *think → tool call → observe* loops, and is trained with **Group Relative Policy Optimization (GRPO)** on top of [verl](https://github.com/volcengine/verl).

Supported benchmarks (download + preprocess scripts included):

| Dataset        | Hugging Face repo              | Config name   |
|----------------|--------------------------------|---------------|
| LongVideoBench | `lmms-lab/LongVideoBench`      | `lvb`         |
| LVBench        | `lmms-lab/LVBench`             | `lvbench`     |
| Video-MME      | `lmms-lab/Video-MME`           | `videomme`    |
| MLVU           | `MLVU/MVLU`                    | `mlvu`        |
| CGBench        | (local / custom)               | `cgbench`     |

---

## Requirements

- Linux + NVIDIA GPU (CUDA **12.4+** recommended)
- Python **≥ 3.11**
- Conda or [uv](https://github.com/astral-sh/uv)
- Sufficient disk for video datasets (hundreds of GB)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/UTAustin-SwarmLab/VSeek-R1.git
cd VSeek-R1
git submodule update --init --recursive   # pulls vendor/verl
```

### 2. Create an environment

**Option A — uv (recommended on a workstation):**

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
uv sync
```

**Option B — conda:**

```bash
conda create -n vseek python=3.11 -y
conda activate vseek
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 3. Install CUDA build deps + FlashAttention

```bash
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

# Choose one inference stack
pip install -r requirements_vllm.txt      # vLLM (default for training / eval)
# or
pip install -r requirements_sglang.txt    # SGLang

pip install flash-attn==2.8.3 --no-build-isolation
```

### 4. Install VSeek + verl

```bash
pip install -e .
pip install -e vendor/verl
```

### 5. Sanity check

```bash
python -c "import torch; import vllm; print(torch.__version__, torch.cuda.is_available())"
```

> **HPC / Slurm users:** do **not** follow this path blindly on TACC Vista (Grace Hopper / ARM). Use [`README_SLURM.md`](README_SLURM.md) instead — vLLM and FlashAttention must be built for `sm_90`.

---

## Download Datasets

Set a root directory and download with `huggingface-cli` (faster than the Python helper):

```bash
export DATASET_ROOT=/path/to/datasets   # e.g. /nas/mars/dataset
mkdir -p "$DATASET_ROOT"

huggingface-cli download MLVU/MVLU --repo-type dataset \
  --local-dir "$DATASET_ROOT/MLVU"

huggingface-cli download lmms-lab/Video-MME --repo-type dataset \
  --local-dir "$DATASET_ROOT/Video-MME"

huggingface-cli download lmms-lab/LVBench --repo-type dataset \
  --local-dir "$DATASET_ROOT/LVBench"

huggingface-cli download lmms-lab/LongVideoBench --repo-type dataset \
  --local-dir "$DATASET_ROOT/longvideobench"
```

Or download everything in one loop:

```bash
for dataset in \
  "MLVU/MVLU:MLVU" \
  "lmms-lab/Video-MME:Video-MME" \
  "lmms-lab/LVBench:LVBench" \
  "lmms-lab/LongVideoBench:longvideobench"
do
  IFS=':' read -r repo_id local_dir <<< "$dataset"
  echo "→ $repo_id → $DATASET_ROOT/$local_dir"
  huggingface-cli download "$repo_id" --repo-type dataset \
    --local-dir "$DATASET_ROOT/$local_dir"
done
```

Also place ViCLIP weights somewhere accessible and point configs at them (defaults live under `src/vseek/setting.py`):

- `ViClip-InternVid-10M-FLT.pth`
- `bpe_simple_vocab_16e6.txt.gz`

---

## Process Datasets

Each dataset follows the same pattern:

1. Unzip videos / burn subtitles (when needed)
2. Convert parquet → JSON (Video-MME, LVBench)
3. Build a retrieval index (`run_data_pipeline.py`)
4. Optionally extract PULS primitives
5. Create train/test parquet for RL (`preprocess_*.sh`)

Set shared paths once:

```bash
export DATASET_ROOT=/path/to/datasets
export INDEX_PATH=/path/to/vseek/indexes   # FAISS / frame indexes
export WINDOW_SIZE=8
export PROMPT_TYPE=tagsummary              # or: tag
```

### LongVideoBench (`lvb`)

```bash
# Burn subtitles into frames
python3 scripts/utils/burn_subtitles.py \
  --json-file lvb_val.json \
  --output-dir "$DATASET_ROOT/longvideobench/burn-subtitles" \
  --data-folder "$DATASET_ROOT/longvideobench/LongVideoBench/"

# Index windows
python3 scripts/data_ops/run_data_pipeline.py \
  retriever.window_size=$WINDOW_SIZE \
  dataset.name=lvb \
  dataset.lvb.dataset_path="$DATASET_ROOT/longvideobench/LongVideoBench/" \
  dataset.lvb.burned_path="$DATASET_ROOT/longvideobench/" \
  retriever.index_path="$INDEX_PATH"

# Optional PULS extraction
python3 src/vseek/puls/primitives.py dataset.name=lvb

# RL parquet splits
bash scripts/data_ops/preprocess_lvb.sh \
  --local_dataset_path "$DATASET_ROOT/longvideobench/LongVideoBench" \
  --burned_path "$DATASET_ROOT/longvideobench/" \
  --train_ratio 0.8 \
  --local_save_dir "$DATASET_ROOT/longvideobench" \
  --index_path "$INDEX_PATH" \
  --window_size $WINDOW_SIZE \
  --prompt_type $PROMPT_TYPE
```

### LVBench (`lvbench`)

```bash
CHUNKS_DIR="$DATASET_ROOT/LVBench/video_chunks" \
OUTPUT_DIR="$DATASET_ROOT/LVBench/videos" \
  bash scripts/utils/unzip_videos.sh

python3 scripts/utils/convert_parquet.py \
  --parquet_path "$DATASET_ROOT/LVBench/data/train-00000-of-00001.parquet" \
  --json_path "$DATASET_ROOT/LVBench/data/lvbench_val.json"

python3 scripts/data_ops/run_data_pipeline.py \
  retriever.window_size=$WINDOW_SIZE \
  dataset.name=lvbench \
  dataset.lvbench.dataset_path="$DATASET_ROOT/LVBench" \
  dataset.lvbench.burned_path="$DATASET_ROOT/LVBench/videos" \
  retriever.index_path="$INDEX_PATH"

python3 src/vseek/puls/primitives.py dataset.name=lvbench

bash scripts/data_ops/preprocess_lvbench.sh \
  --local_dataset_path "$DATASET_ROOT/LVBench" \
  --burned_path "$DATASET_ROOT/LVBench/videos" \
  --train_ratio 0.8 \
  --local_save_dir "$DATASET_ROOT/LVBench" \
  --index_path "$INDEX_PATH" \
  --window_size $WINDOW_SIZE \
  --prompt_type $PROMPT_TYPE
```

### Video-MME (`videomme`)

```bash
SUBTITLE_DIR="$DATASET_ROOT/Video-MME" \
CHUNKS_DIR="$DATASET_ROOT/Video-MME" \
OUTPUT_DIR="$DATASET_ROOT/Video-MME/videos" \
  bash scripts/utils/unzip_videos.sh

python3 scripts/utils/convert_parquet.py \
  --parquet_path "$DATASET_ROOT/Video-MME/videomme/test-00000-of-00001.parquet" \
  --json_path "$DATASET_ROOT/Video-MME/videomme/videomme_val.json"

python3 scripts/utils/burn_subtitles.py \
  --json-file videomme/videomme_val.json \
  --output-dir "$DATASET_ROOT/Video-MME/burn-subtitles" \
  --data-folder "$DATASET_ROOT/Video-MME"

python3 scripts/data_ops/run_data_pipeline.py \
  dataset.name=videomme \
  retriever.window_size=$WINDOW_SIZE \
  dataset.videomme.dataset_path="$DATASET_ROOT/Video-MME/" \
  dataset.videomme.burned_path="$DATASET_ROOT/Video-MME/burn-subtitles/" \
  retriever.index_path="$INDEX_PATH" \
  retriever.gpu_number=0

python3 src/vseek/puls/primitives.py dataset.name=videomme

bash scripts/data_ops/preprocess_videomme.sh \
  --local_dataset_path "$DATASET_ROOT/Video-MME" \
  --burned_path "$DATASET_ROOT/Video-MME/burn-subtitles" \
  --train_ratio 0.8 \
  --local_save_dir "$DATASET_ROOT/Video-MME" \
  --index_path "$INDEX_PATH" \
  --window_size $WINDOW_SIZE \
  --prompt_type $PROMPT_TYPE
```

### MLVU (`mlvu`)

Tasks: `plotQA`, `needle`, `ego`, `count`, `order`, `anomaly_reco`, `topic_reasoning`.

```bash
python3 scripts/data_ops/run_data_pipeline.py \
  retriever.window_size=$WINDOW_SIZE \
  dataset.name=mlvu \
  dataset.mlvu.dataset_path="$DATASET_ROOT/MLVU/MLVU" \
  dataset.mlvu.burned_path="$DATASET_ROOT/MLVU/MLVU" \
  retriever.index_path="$INDEX_PATH"

python3 src/vseek/puls/primitives.py dataset.name=mlvu

bash scripts/data_ops/preprocess_mlvu.sh \
  --local_dataset_path "$DATASET_ROOT/MLVU/MLVU" \
  --burned_path "$DATASET_ROOT/MLVU/MLVU" \
  --train_ratio 0.8 \
  --local_save_dir "$DATASET_ROOT/MLVU" \
  --index_path "$INDEX_PATH" \
  --window_size $WINDOW_SIZE \
  --prompt_type $PROMPT_TYPE
```

### CGBench (`cgbench`)

```bash
SUBTITLE_DIR="$DATASET_ROOT/CGBench" \
CHUNKS_DIR="$DATASET_ROOT/CGBench" \
OUTPUT_DIR="$DATASET_ROOT/CGBench/videos" \
  bash scripts/utils/unzip_videos.sh

python3 scripts/utils/burn_subtitles.py \
  --json-file cgbench_mini.json \
  --output-dir "$DATASET_ROOT/CGBench/burn-subtitles" \
  --data-folder "$DATASET_ROOT/CGBench"

python3 scripts/data_ops/run_data_pipeline.py \
  dataset.name=cgbench \
  retriever.window_size=$WINDOW_SIZE \
  dataset.cgbench.dataset_path="$DATASET_ROOT/CGBench/" \
  dataset.cgbench.burned_path="$DATASET_ROOT/CGBench/burn-subtitles/" \
  retriever.index_path="$INDEX_PATH"

bash scripts/data_ops/preprocess_cgbench.sh \
  --local_dataset_path "$DATASET_ROOT/CGBench" \
  --burned_path "$DATASET_ROOT/CGBench/burn-subtitles" \
  --train_ratio 0.8 \
  --local_save_dir "$DATASET_ROOT/CGBench" \
  --index_path "$INDEX_PATH" \
  --window_size $WINDOW_SIZE \
  --prompt_type $PROMPT_TYPE
```

---

## Run the Retriever Server

The agent calls a Flask retrieval server during training and inference.

On **local** setups (`v0.6.1`), defaults in `config.yaml` / `toolconfig.yaml` use `/nas/mars/...` datasets and `/home/.../vseek/dataset` indexes (port **9005**). On **TACC**, those must be rewritten to `$SCRATCH` / `$WORK` paths (port **9000**) — see [Path conventions](#path-conventions-local-vs-tacc) and [`README_SLURM.md`](README_SLURM.md).

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/vseek/tools/server.py \
  retriever.window_size=$WINDOW_SIZE \
  dataset.name=['lvb','lvbench','videomme','mlvu'] \
  retriever.index_path="$INDEX_PATH" \
  retriever.retrieval_model_path=/path/to/ViClip-InternVid-10M-FLT.pth \
  retriever.text_encoder_model_path=/path/to/bpe_simple_vocab_16e6.txt.gz
```

Update `src/vseek/config/retriever/toolconfig.yaml` so the agent points at this server URL **and** the same dataset/index roots the server was started with.

---

## Inference

### Uniform frame baseline

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_uniform_agent_data.py \
  inference.max_images_per_turn=64 \
  +inference.agent_prompt_type=cot \
  llm.model=Qwen/Qwen3-VL-4B-Thinking \
  inference.output_dir=results/uniform_q3_4bt_cot \
  inference.max_output_tokens=2048 \
  inference.gpu_number=0
```

### VSeek agent (vLLM)

```bash
bash scripts/evals/run_vseek.sh
# or see scripts/run_agent_data_vllm.py / scripts/evals/*
```

### Merge FSDP checkpoints → Hugging Face format

```bash
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir checkpoints/vseek/<exp>/global_step_XXX/actor \
  --target_dir checkpoints/vseek/<exp>/global_step_XXX/actor/huggingface
```

---

## Training (GRPO)

Example single-node GRPO run (edit paths / GPU count to match your machine):

```bash
bash scripts/train/vseek_grpo_tag_qvl34Bt_vllm_tagsummary.sh
```

Key knobs live in:

- `src/vseek/config/lvb_grpo.yaml` — trainer / rollout config
- `src/vseek/config/retriever/toolconfig.yaml` — tool server
- `src/vseek/config/retriever/vllm_agent.yaml` — agent loop

Multi-node Slurm jobs: [`README_SLURM.md`](README_SLURM.md).

---

## Evaluation

Convenience wrappers under `scripts/evals/`:

```bash
bash scripts/evals/run_vseek.sh
bash scripts/evals/run_uniform_vllm.sh
bash scripts/evals/run_gpt_eval.sh
bash scripts/evals/vllm_vseek_tests_ckpt.sh \
  --devices 0 \
  --batch-size 16 \
  --output-base results/vseek/my-run \
  --model-path /path/to/hf/checkpoint \
  --datasets lvb,lvbench,videomme,mlvu
```

---

## Repository Layout

```text
VSeek-R1/
├── src/vseek/           # Package: agent, tools, trainer, data, PULS
├── scripts/
│   ├── data_ops/        # Indexing + parquet preprocessing
│   ├── train/           # GRPO launch scripts
│   ├── evals/           # Evaluation wrappers
│   ├── utils/           # Download, unzip, burn subtitles, …
│   └── tacc/            # Slurm / Ray helpers (see README_SLURM.md)
├── vendor/verl/         # Vendored verl (RL training)
├── requirements_vllm.txt
├── requirements_sglang.txt
├── requirements_vllm_slurm2.txt
├── README.md            # This file (workstation setup)
└── README_SLURM.md      # TACC / Slurm installation
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{vseekr1,
  title     = {{VSeek-R1}: TODO — fill in camera-ready title},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2026}
}
```

---

## Acknowledgments

Built on [verl](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), and the public video understanding benchmarks listed above.
