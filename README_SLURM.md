# VSeek-R1 — Slurm / TACC Installation

HPC setup for **TACC Vista** (Grace Hopper, ARM64, compute capability **sm_90**).  
For workstation / x86 installs, use the main [`README.md`](README.md).

**Working stack:** PyTorch **2.10.0** · CUDA **12.8** · vLLM **0.12.0** (built from source) · conda env `vseek-vllm4` · Python **3.12**

This matches `scripts/tacc/run_vseek_job.slurm` and `scripts/tacc/ray_wrapper.sh`.

> **Path note:** `harsh_tacc` / these Slurm scripts do **not** use the local `v0.6.1` roots (`/nas/mars/...`, `/home/.../vseek/dataset`). Datasets, indexes, ViCLIP weights, and training parquet live under TACC **`$SCRATCH`** / **`$WORK`**. See [Path conventions](#path-conventions-tacc-vs-local) before starting the retrieval server.

---

## Table of Contents

- [Path conventions (TACC vs local)](#path-conventions-tacc-vs-local)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Build verification](#build-verification)
- [Install VSeek + verl](#install-vseek--verl)
- [Paths on Vista](#paths-on-vista)
- [Launching jobs](#launching-jobs)
- [Retriever server on a compute node](#retriever-server-on-a-compute-node)
- [Troubleshooting](#troubleshooting)

---

## Path conventions (TACC vs local)

`v0.6.1` was developed against a **local lab NAS**. The `harsh_tacc` branch retargets **retriever YAML** and adds **`scripts/tacc/`**, but many other scripts still embed `/nas/mars/...`. If you sync code across branches without auditing paths, the retriever server and training jobs will look in the wrong place.

### Branch delta (paths that actually change)

| File | Local (`v0.6.1`) | TACC (`harsh_tacc`) |
|------|------------------|---------------------|
| `src/vseek/config/retriever/config.yaml` | `/nas/mars/dataset/...`, index `/home/.../vseek/dataset`, ViCLIP `/nas/mars/model_weights/...`, port **9005** | `$SCRATCH/.../datasets/...`, index `$SCRATCH/.../vseek`, ViCLIP `$SCRATCH/.../model_weights/...`, port **9000** |
| `src/vseek/config/retriever/toolconfig.yaml` | same local pattern (MLVU path is `.../MLVU` without the nested `MLVU/MLVU` used in `config.yaml`) | `$SCRATCH` roots with `.../MLVU/MLVU` |
| `scripts/tacc/*` | absent / unused | parquet + checkpoints under `$WORK/.../vista/...`; `PROJECT_ROOT` → `/home1/.../VSeek-R1`; `HF_HOME` → `$WORK` or `$SCRATCH` |

| Resource | Local (`v0.6.1`) | TACC (`harsh_tacc`) |
|----------|------------------|---------------------|
| Raw / burned videos | `/nas/mars/dataset/<name>/...` | `/scratch/<alloc>/<user>/datasets/<name>/...` |
| FAISS / frame indexes | `/home/<user>/vseek/dataset` | `/scratch/<alloc>/<user>/vseek` |
| ViCLIP weights | `/nas/mars/model_weights/viclip/` | `/scratch/<alloc>/<user>/model_weights/viclip/` |
| Train/val parquet (Slurm entrypoints) | `/nas/mars/vseek/...` or `.../window_8/` | `/work/<alloc>/<user>/vista/vseek/tagsummary/{train,test}.parquet` |
| Checkpoints | local `checkpoints/` | `/work/.../vista/checkpoints` or `${SCRATCH}/checkpoints` |
| Retriever port | `9005` | `9000` |

Example mapping used on Vista development (replace with your allocation / username):

```text
# Local (v0.6.1)
/nas/mars/dataset/Video-MME
/home/hg22723/vseek/dataset
/nas/mars/model_weights/viclip/...

# SLURM (harsh_tacc)
/scratch/11123/harshgoel99/datasets/Video-MME
/scratch/11123/harshgoel99/vseek
/scratch/11123/harshgoel99/model_weights/viclip/...
/work/11123/harshgoel99/vista/vseek/tagsummary/train.parquet
```

### Not rewritten on `harsh_tacc` (still `/nas/mars` if you call them)

Do **not** assume every script is Vista-ready just because you are on the Slurm branch:

- `src/vseek/setting.py` — unchanged local NAS defaults
- `scripts/train/*.sh` — still `/nas/mars/...` parquet paths (use `scripts/tacc/run_vseek_job.slurm` or override `data.train_files` / `data.val_files`)
- `scripts/data_ops/preprocess_*.sh`, `scripts/utils/unzip_videos.sh` — NAS defaults for dataset / index / unzip dirs
- `scripts/evals/vllm_vseek_tests_ckpt.sh` — NAS `window_8` parquet roots
- Assorted local `PROJECT_ROOT=/home/hg22723/projects/VSeek-R1` hardcodes in Python launchers

### Inconsistencies inside the TACC layout itself

- Notes on `harsh_tacc` sometimes list data/indexes under **`$WORK/.../vista/{datasets,vseek}`**, while the committed YAML uses **`$SCRATCH/.../{datasets,vseek}`**. Use whichever tree you actually copied data into, and keep YAML + server CLI aligned.
- `run_vseek_job*.slurm` hard-codes `/work/11123/harshgoel99/...` for checkpoints/parquet; `run_vseek_all_jobs*.slurm` prefers `${SCRATCH}/checkpoints` + env `TRAIN_DATA_PATH`.
- `HF_HOME` is `$WORK/huggingface` in some scripts and `$SCRATCH/huggingface` in `ray_wrapper.sh` / all-jobs scripts — pick one cache and stick to it on workers.

Update (or Hydra-override) before launching:

- `src/vseek/config/retriever/config.yaml` — `retriever.index_path`, `retrieval_model_path`, `dataset.*.dataset_path` / `burned_path`
- `src/vseek/config/retriever/toolconfig.yaml` — nested `dataset` / `retriever` blocks (keep in sync with the server)
- `scripts/tacc/run_vseek_job.slurm` — `data.train_files`, `data.val_files`, `CHECKPOINT_DIR`, `PROJECT_ROOT` fallback
- `scripts/tacc/vseek_grpo_tag_qvl34Bt_vllm_tagsummary.sh` — parquet + `retriever.server_url`
- Any non-`tacc` train / preprocess / eval script you still run on Vista

---

## Prerequisites

- TACC account with Vista (`gh` partition) access
- Conda / Miniforge under `$WORK` or `$HOME`
- Space under `$WORK` / `$SCRATCH` for Hugging Face caches and builds
- This repo cloned (with `vendor/verl` submodule)

```bash
git clone https://github.com/UTAustin-SwarmLab/VSeek-R1.git
cd VSeek-R1
git submodule update --init --recursive
```

Run the build steps on a **compute node** (or interactive GPU allocation), not the login node.

---

## Install

**Target:** `gcc/14.2.0` · CUDA 12.8 · Torch 2.10.0 (cu128) · vLLM 0.12.0 from source · FlashAttention 2.8.3 · flashinfer 0.5.3 · `numpy<2` · `TORCH_CUDA_ARCH_LIST=9.0`

### 1. Modules + conda env

```bash
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"

conda create --name vseek-vllm4 python=3.12 -y
conda activate vseek-vllm4
module load gcc/14.2.0 cuda/12.8   # reload after activate if needed
```

### 2. PyTorch 2.10.0 (cu128) + flashinfer

```bash
pip install torch==2.10.0 torchvision torchaudio flashinfer-python \
  --index-url https://download.pytorch.org/whl/cu128 \
  --no-cache-dir

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Build vLLM 0.12.0 from source

```bash
cd "$WORK"
rm -rf vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout releases/v0.12.0

module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="sm_90"

echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC (expect gcc 14.2)"
$CC --version | head -1

# Clean prior build artifacts
rm -rf .deps/ build/ vllm/*.so
find vllm -name '*.so' -delete 2>/dev/null || true

python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/common.txt

cat > /tmp/constraints.txt << 'EOF'
numpy<2
EOF

MAX_JOBS=24 pip install -e . \
  --no-build-isolation \
  --no-cache-dir \
  --constraint /tmp/constraints.txt \
  --verbose
```

Re-pin Torch / flashinfer after the vLLM install (pip may pull other versions as deps):

```bash
pip install torch==2.10.0 torchvision torchaudio flashinfer-python \
  --index-url https://download.pytorch.org/whl/cu128 \
  --no-cache-dir
```

### 4. Training / inference Python deps

```bash
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
  "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
  "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 \
  liger-kernel mathruler pytest py-spy pre-commit ruff tensorboard \
  --no-cache-dir

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
  "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" \
  --no-cache-dir
```

### 5. FlashAttention + flashinfer + OpenCV

```bash
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export TORCH_CUDA_ARCH_LIST="9.0"

FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.8.3 \
  --no-build-isolation \
  --no-cache-dir \
  --verbose

pip install flashinfer-python==0.5.3 --no-cache-dir
pip install opencv-python
pip install opencv-fixer && python -c "from opencv_fixer import AutoFix; AutoFix()"
```

---

## Build verification

Run on a **GPU compute node** with `vseek-vllm4` active:

```bash
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export TORCH_CUDA_ARCH_LIST="9.0"

python - <<'PY'
import torch, numpy as np
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"NumPy: {np.__version__}")
assert torch.__version__.startswith("2.10"), torch.__version__
assert "cu128" in torch.__version__ or torch.version.cuda.startswith("12.8")
assert int(np.__version__.split(".")[0]) < 2
print("OK: Torch 2.10 + cu128 + NumPy < 2")
PY

python -c "import vllm; print('vLLM', vllm.__version__)"
```

Confirm compiled extensions include **sm_90**:

```bash
python - <<'PY'
import os, subprocess, vllm

def check(*rel):
    root = os.path.dirname(vllm.__file__)
    for r in rel:
        path = os.path.join(root, r)
        if not os.path.isfile(path):
            print(f"[SKIP] missing {path}")
            continue
        out = subprocess.check_output(["strings", path], text=True, errors="ignore")
        sms = sorted({t for t in out.split() if t.startswith("sm_")})
        ok = any(s.startswith("sm_90") for s in sms)
        print(("[PASS]" if ok else "[FAIL]"), os.path.basename(path), sms)

check("_C.abi3.so",
      "vllm_flash_attn/_vllm_fa2_C.abi3.so",
      "vllm_flash_attn/_vllm_fa3_C.abi3.so")
PY

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
```

`libstdc++` should expose **GLIBCXX_3.4.32+** (needed by FlashAttention):

```bash
strings /lib64/libstdc++.so.6 | grep GLIBCXX | tail -3
```

---

## Install VSeek + verl

```bash
cd "$HOME/VSeek-R1"   # or your clone path
conda activate vseek-vllm4
module load gcc/14.2.0 cuda/12.8

pip install -r requirements_vllm_slurm2.txt --no-deps
pip install -e vendor/verl
pip install -e .
```

`requirements_vllm_slurm2.txt` deliberately **omits** Torch / vLLM pins so the source builds above are not overwritten (`--no-deps`).

Optional: build **decord** from source if the pip wheel fails on ARM:

```bash
conda install 'ffmpeg=4' -y
# clone decord, then:
mkdir -p build && cd build
CC=$(which gcc) CXX=$(which g++) cmake .. \
  -DUSE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DFFMPEG_DIR="$CONDA_PREFIX"
make -j"$(nproc)"
cd ../python && python setup.py install
```

---

## Paths on Vista

Typical layout used in development (adjust to your allocation). **Do not reuse `/nas/mars/...` from `v0.6.1` on Vista.**

| Resource | Example path |
|----------|----------------|
| Raw / burned datasets | `$SCRATCH/.../datasets/` |
| Indexes / processed VSeek data | `$SCRATCH/.../vseek` (also parquet under `$WORK/.../vista/vseek`) |
| Hugging Face cache | `$SCRATCH/huggingface` or `$WORK/huggingface` |
| ViCLIP weights | `$SCRATCH/.../model_weights/viclip/` |
| Checkpoints | `$WORK/.../vista/checkpoints` |

Export caches in job scripts so workers share the same hub:

```bash
export HF_HOME="${SCRATCH}/huggingface"   # or ${WORK}/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
```

---

## Launching jobs

Slurm helpers live under `scripts/tacc/`. They assume conda env **`vseek-vllm4`** and modules **`gcc/14.2.0` + `cuda/12.8`**.

| Script | Purpose |
|--------|---------|
| `run_vseek_job.slurm` | Multi-node Ray + GRPO (this stack) |
| `ray_wrapper.sh` | Env wrapper invoked on each Ray worker |
| `vseek_grpo_tag_qvl34Bt_vllm_tagsummary.sh` | Training command payload |
| `run_vseek_all_jobs*.slurm` | Batch eval / multi-job launchers |

Submit:

```bash
cd "$HOME/VSeek-R1"
sbatch scripts/tacc/run_vseek_job.slurm
```

The job script will:

1. Load `gcc/14.2.0` + `cuda/12.8` and activate `vseek-vllm4`
2. Resolve the Ray head IP from `$SLURM_JOB_NODELIST`
3. Start Ray head + workers via `ray_wrapper.sh` (SSH, not `ibrun`)
4. Start the retrieval server and launch GRPO training

Edit partition / node count at the top of the `.slurm` file (`#SBATCH -p gh`, `-N`, …) and update dataset / index paths inside the script before submitting.

Worker env tip: dump head-node CUDA-related env and source it on workers (already done by the Slurm scripts):

```bash
env | grep -E '^(PATH|LD_LIBRARY_PATH|CUDA_|NCCL_|CONDA_)' > worker_env.sh
# ray_wrapper.sh sources ~/worker_env.sh on each node
```

---

## Retriever server on a compute node

Pass **TACC** index / weight paths (not `/nas/mars/...` from local `v0.6.1`):

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/vseek/tools/server.py \
  retriever.window_size=8 \
  dataset.name=['lvb','lvbench','videomme','mlvu'] \
  retriever.index_path=/scratch/<alloc>/<user>/vseek \
  retriever.retrieval_model_path=/scratch/<alloc>/<user>/model_weights/viclip/ViClip-InternVid-10M-FLT.pth \
  retriever.text_encoder_model_path=/scratch/<alloc>/<user>/model_weights/viclip/bpe_simple_vocab_16e6.txt.gz
```

Also ensure `config.yaml` / `toolconfig.yaml` dataset `dataset_path` and `burned_path` entries point at `$SCRATCH/.../datasets/...`. Point the Slurm job’s `retriever.server_url` at `http://<head-node-ip>:9000`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `sm_80` / missing `sm_90` in `.so` files | Built without `TORCH_CUDA_ARCH_LIST=9.0` | Clean build dirs; rebuild vLLM / flash-attn with arch env vars |
| Ray workers: `torch.cuda.is_available() == False` | Worker missing modules / wrong `CUDA_HOME` | Ensure `ray_wrapper.sh` loads `gcc/14.2.0`, `cuda/12.8`, and `vseek-vllm4` |
| Torch version drifted after `pip install -e .` (vLLM) | Dep resolver upgraded/downgraded Torch | Re-run the Torch 2.10.0 re-pin step |
| GPU OOM / Ray oversubscription | Too many GPUs claimed per worker | Keep `--num-gpus=1` per Ray process; lower micro-batch sizes |
| `GLIBCXX` symbol errors | Too-old system libstdc++ vs flash-attn | Keep module `gcc/14.2.0` loaded |
| Login-node builds fail | No GPU / wrong toolchain | Build on an interactive `idev` / compute allocation |

---

## Related docs

- Workstation setup & data pipeline: [`README.md`](README.md)
- Training script reference: `scripts/train/vseek_grpo_tag_qvl34Bt_vllm_tagsummary.sh`
- Slurm job entrypoint: `scripts/tacc/run_vseek_job.slurm`
