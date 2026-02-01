Next, install uv:
```bash
pip install -r requirement_vllm.text
```

Next, set up venv:
```bash
uv venv .venv
```


Finally, install everything in `pyproject.toml` to build project dependencies:
```bash
uv sync
```


'''
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 cuda-toolkit -c pytorch -c nvidia
'''

export CUDA_HOME="$CONDA_PREFIX"

pip install -r requirements_sglang.txt

or
pip install -r requirements_vllm.txt
pip install flash-attn==2.8.3 --no-build-isolation
export CUDA_HOME="$CONDA_PREFIX"


''' on arch, try to get the cuda 12.8 librarie by

pip install -r requirements_vllm_slurm2.txt
pip3 install --force torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.0
## apply patches torch>=2.7.0 and license={text="Apache-2.0"} in the project.toml of vllm, delete licnese fuliles

module load gcc/14.2.0
module load cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)


MAX_JOBS=8 VLLM_NO_USAGE_STATS=1 pip install . --no-build-isolation 
--no-deps
'''




'''

cFor tacc



module load gcc/14.2.0
module load cuda/12.8
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME="$TACC_CUDA_DIR"
export TORCH_CUDA_ARCH_LIST="9.0"

MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

<!-- pip install --force-reinstall --no-cache-dir transformers

pip install outlines==1.2.0 outlines-core=0.2.11

pip install cuda-toolkit ninja cudart


export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LIBRARY_PATH}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH}"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# (Optional, but often helpful for CUDA builds if specific compiler is needed)
export PATH=$CONDA_PREFIX/bin:$PATH -->


pip install vendor/verl



## Download Datasets

All datasets are downloaded to `/nas/mars/dataset/` by default. Make sure you have sufficient disk space.

### Option 1: Using huggingface-cli (Recommended - Faster)

```bash
# Download MLVU dataset
huggingface-cli download MLVU/MVLU --repo-type dataset --local-dir /nas/mars/dataset/MLVU

# Download Video-MME dataset
huggingface-cli download lmms-lab/Video-MME --repo-type dataset --local-dir /nas/mars/dataset/Video-MME

# Download LVBench dataset
huggingface-cli download lmms-lab/LVBench --repo-type dataset --local-dir /nas/mars/dataset/LVBench

# Download LongVideoBench dataset
huggingface-cli download lmms-lab/LongVideoBench --repo-type dataset --local-dir /nas/mars/dataset/longvideobench
```

### Option 2: Using Python download script

<!-- ```bash
# Download MLVU dataset
python3 scripts/utils/download.py \
    --dataset_name MLVU/MVLU \
    --download_directory /nas/mars/dataset/MLVU

# Download Video-MME dataset
python3 scripts/utils/download.py \
    --dataset_name lmms-lab/Video-MME \
    --download_directory /nas/mars/dataset/Video-MME

# Download LVBench dataset
python3 scripts/utils/download.py \
    --dataset_name lmms-lab/LVBench \
    --download_directory /nas/mars/dataset/LVBench

# Download LongVideoBench dataset
python3 scripts/utils/download.py \
    --dataset_name lmms-lab/LongVideoBench \
    --download_directory /nas/mars/dataset/longvideobench
``` -->

### Download All Datasets at Once

```bash
# Create base directory
mkdir -p /nas/mars/dataset

# Download all datasets
for dataset in "MLVU/MVLU:MLVU" "lmms-lab/Video-MME:Video-MME" "lmms-lab/LVBench:LVBench" "lmms-lab/LongVideoBench:longvideobench"; do
    IFS=':' read -r repo_id local_dir <<< "$dataset"
    echo "Downloading $repo_id to /nas/mars/dataset/$local_dir"
    huggingface-cli download "$repo_id" --repo-type dataset --local-dir "/nas/mars/dataset/$local_dir"
done
```

## Process Datasets

### LongVideoBench

#### Burn Subtitles

```bash
python3 scripts/utils/burn_subtitles.py --json-file lvb_val.json --output-dir /nas/mars/dataset/longvideobench/burn-subtitles --data-folder /nas/mars/dataset/longvideobench/LongVideoBench/
```

#### Run indexing to target window size

```bash
python3 scripts/data_ops/run_data_pipeline.py \
    retriever.window_size=8 \
    dataset.name='lvb' \
    dataset.lvb.dataset_path="/nas/mars/dataset/longvideobench/LongVideoBench/" \
    dataset.lvb.burned_path="/nas/mars/dataset/longvideobench/" \
    retriever.index_path="/home/hg22723/vseek/dataset"
```

```bash
 python3 src/vseek/puls/primitives.py dataset.name='lvb'
```
#### Run data preprocessing to create train and test parquets for RL

python3
```bash
bash scripts/data_ops/preprocess_lvb.sh \
    --local_dataset_path "/nas/mars/dataset/longvideobench/LongVideoBench" \
    --burned_path "/nas/mars/dataset/longvideobench/" \
    --train_ratio 0.8 \
    --local_save_dir /nas/mars/dataset/longvideobench \
    --index_path /home/hg22723/vseek/dataset \
    --window_size 8 \
    --prompt_type tagsummary
```
# tag and tagsummary

### LVBench 
#### Unzip Videos and convert parquet

```bash
CHUNKS_DIR=/nas/mars/dataset/LVBench/video_chunks OUTPUT_DIR=/nas/mars/dataset/LVBench/videos bash scripts/utils/unzip_videos.sh 
```

```bash
python3 scripts/utils/convert_parquet.py --parquet_path /nas/mars/dataset/LVBench/data/train-00000-of-00001.parquet --json_path /nas/mars/dataset/LVBench/data/lvbench_val.json
```

#### Run indexing to target window size
```bash
python3 scripts/data_ops/run_data_pipeline.py \
    retriever.window_size=8 \
    dataset.name='lvbench' \
    dataset.lvbench.dataset_path="/nas/mars/dataset/LVBench" \
    dataset.lvbench.burned_path="/nas/mars/dataset/LVBench/videos" \
    retriever.index_path="/home/hg22723/vseek/dataset"
```
```bash
 python3 src/vseek/puls/primitives.py dataset.name='lvbench'
```
#### Run data preprocessing to create train and test parquets for RL

python3
```bash
bash scripts/data_ops/preprocess_lvbench.sh \
    --local_dataset_path "/nas/mars/dataset/LVBench" \
    --burned_path "/nas/mars/dataset/LVBench/videos" \
    --train_ratio 0.8 \
    --local_save_dir /nas/mars/dataset/LVBench \
    --index_path /home/hg22723/vseek/dataset \
    --window_size 8 \
    --prompt_type tagsummary
```

### Video-MME

#### Unzip Videos and convert parquet
```bash
SUBTITLE_DIR=/nas/mars/dataset/Video-MME CHUNKS_DIR=/nas/mars/dataset/Video-MME OUTPUT_DIR=/nas/mars/dataset/Video-MME/videos bash scripts/utils/unzip_videos.sh
```


```bash
python3 scripts/utils/convert_parquet.py --parquet_path /nas/mars/dataset/Video-MME/videomme/test-00000-of-00001.parquet --json_path /nas/mars/dataset/Video-MME/videomme/videomme_val.json
```

#### Burn Subtitles

```bash
python3 scripts/utils/burn_subtitles.py --json-file videomme/videomme_val.json --output-dir /nas/mars/dataset/Video-MME/burn-subtitles --data-folder /nas/mars/dataset/Video-MME

```

#### Run indexing to target window size

```bash
    python3 scripts/data_ops/run_data_pipeline.py \
    dataset.name=videomme \
    retriever.window_size=8 \
    dataset.videomme.dataset_path=/nas/mars/dataset/Video-MME/ \
    dataset.videomme.burned_path=/nas/mars/dataset/Video-MME/burn-subtitles/ \
    retriever.index_path=/home/hg22723/vseek/dataset \
    retriever.gpu_number=0
```
```bash
 python3 src/vseek/puls/primitives.py dataset.name='videomme'
```

#### Run data preprocessing to create train and test parquets for RL

python3
```bash
bash scripts/data_ops/preprocess_videomme.sh \
    --local_dataset_path "/nas/mars/dataset/Video-MME" \
    --burned_path "/nas/mars/dataset/Video-MME/burn-subtitles" \
    --train_ratio 0.8 \
    --local_save_dir /nas/mars/dataset/Video-MME \
    --index_path /home/hg22723/vseek/dataset \
    --window_size 8 \
    --prompt_type tagsummary
```

### MLVU
plotQA, needle, ego, count, order, anomaly_reco, topic_reasoning
```bash
# Process MLVU dataset (if preprocessor exists)
# Add your MLVU processing commands here
```
```bash
python3 scripts/data_ops/run_data_pipeline.py \
    retriever.window_size=8 dataset.name='mlvu' \
    dataset.mlvu.dataset_path="/nas/mars/dataset/MLVU/MLVU" \
    dataset.mlvu.burned_path="/nas/mars/dataset/MLVU/MLVU" \
    retriever.index_path="/home/hg22723/vseek/dataset"
```

#### Run data preprocessing to create train and test parquets for RL

```bash
 python3 src/vseek/puls/primitives.py dataset.name='mlvu'
```

python3
```bash
bash scripts/data_ops/preprocess_mlvu.sh \
    --local_dataset_path "/nas/mars/dataset/MLVU/MLVU" \
    --burned_path "/nas/mars/dataset/MLVU/MLVU" \
    --train_ratio 0.8 \
    --local_save_dir /nas/mars/dataset/MLVU \
    --index_path /home/hg22723/vseek/dataset \
    --window_size 8 \
    --prompt_type tagsummary

# Option 2: Use the convenience script
bash scripts/preprocess_mvlu.sh
```




### For Video-MME and LVBench 





### For VideoMME and LVBench

the dowloaded test file is a test.parquet, we extract it to .json
python3 scripts/utils/convert_parquet.py --parquet_path /nas/mars/dataset/LVBench/data/train-00000-of-00001.parquet --json_path /nas/mars/dataset/LVBench/data/lvbench_val.json

python3 scripts/utils/convert_parquet.py --parquet_path /nas/mars/dataset/Video-MME/videomme/test-00000-of-00001.parquet --json_path /nas/mars/dataset/Video-MME/videomme/videomme_val.json

### Brun subtitles for Video MME and LongVideoBench
python3 scripts/utils/burn_subtitles.py --json-file videomme/videomme_val.json --output-dir /nas/mars/dataset/Video-MME/burn-subtitles --data-folder /nas/mars/dataset/Video-MME

python3 scripts/utils/burn_subtitles.py --json-file lvb_val.json --output-dir /nas/mars/experiment_result/test --data-folder /nas/mars/dataset/longvideobench/LongVideoBench/

### LongVideoBench (LVB)

python3 scripts/data_ops/run_data_pipeline.py dataset.name=lvbench retriever.window_size=8 da
taset.dataset_path=/nas/mars/dataset/LVBench dataset.burned_path=/nas/mars/dataset/LVBench/videos/ retriever.index_path=/home/hg22723/vseek/dataset


```bash
# Process LongVideoBench dataset
python3 scripts/data_ops/run_data_pipeline.py \
    retriever.window_size=8 \
    dataset.name='lvb' \
    dataset.dataset_path="/nas/mars/dataset/longvideobench/LongVideoBench/" \
    dataset.burned_path="/nas/mars/dataset/longvideobench/"
```

### LVBench

```bash
# Option 1: Run full pipeline (index + parquet creation)
python src/data/lvbench_preprocessor.py \
    --local_dataset_path /nas/mars/dataset/LVBench \
    --local_save_dir ~/data/lvbench \
    --index_path /nas/mars/dataset/lvbench_index \
    --window_size 8 \
    --embed_frames \
    --prompt_type tag \
    --retrieval_model_path /path/to/viclip \
    --gpu_number 0 \
    --mode both

# Option 2: Use the convenience script
bash scripts/preprocess_lvbench.sh
```




python3 scripts/utils/download.py --dataset_name lmms-lab/LongVideoBench --download_directory /nas/mars/dataset/longvideobench -->

huggingface-cli download MLVU/MVLU --repo-type dataset --local-dir /nas/mars/dataset/MLVU

# Some scripts
python3 scripts/data_ops/run_data_pipeline.py retriever.window_size=8 dataset.name='lvb' dataset.dataset_path="/nas/mars/dataset/longvideobench/LongVideoBench/" dataset.burned_path="/nas/mars/dataset/longvideobench/"


## Run unofrm agent

CUDA_VISIBLE_DEVICES=5 python3 scripts/run_uniform_agent_data.py inference.max_images_per_turn=64 +inference.agent_prompt_type=cot llm.model=Qwen/Qwen3-VL-4B-Thinking inference.output_dir=results/uniform_q3_4bt_cot2 inference.max_output_tokens=2048 inference.gpu_number=5


python3 -m verl.model_merger merge --backend fsdp --local_dir checkpoints/vseek/qwen3-4bt_vl_lvb-emreward-vllm-wtool-tag/global_step_260/actor  --target_dir checkpoints/vseek/qwen3-4bt_vl_lvb-emreward-vllm-wtool-tag/global_step_260/actor/huggingface


### TACC build experimental with cuda 12.7 and custom built vllm (does not work)

CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install -r requirements_vllm.txt # For xformers

install decord from source 
- install ffmpeg=4.4.2 with conda install 'ffmpeg=4'

cd to decord

mdkir build & cd build
C=$(which gcc) CXX=$(which g++) cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=$WORK/miniforge3/envs/vseek-vllm
make

cd python
python setup.py install


### 2.7.1 build


pip3 install torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-build-isolation

module load gcc/13.2.0
module load cuda/12.8
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME="$TACC_CUDA_DIR"
export TORCH_CUDA_ARCH_LIST="9.0"

MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.0
## apply patches torch>=2.7.0 and license={text="Apache-2.0"} in the project.toml of vllm, delete licnese fuliles

export CUDA_HOME="/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8"
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"


CMAKE_ARGS="-DCUDAToolkit_ROOT=$CUDA_HOME -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME" MAX_JOBS=8 VLLM_NO_USAGE_STATS=1 pip install . --no-build-isolation 
--no-deps


### TACC build with torch 2.9, compiles but verl runs into issues

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir


# Test Torch
# python -c "import torch; print(torch.cuda.is_available())"

pip install vllm==0.12.0 --no-cache-dir
# Test vLLM
# vllm serve Qwen/Qwen2.5-1.5B-Instruct
# curl http://localhost:8000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "Qwen/Qwen2.5-1.5B-Instruct",
#         "prompt": "San Francisco is a",
#         "max_tokens": 7,
#         "temperature": 0
#     }'

FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir



#### 2.7.1 cuda 12.9

pip install vllm==0.10.0 --no-cache-dir
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
# Test Torch
# python -c "import torch; print(torch.cuda.is_available())"

# Test vLLM
# vllm serve Qwen/Qwen2.5-1.5B-Instruct
# curl http://localhost:8000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "Qwen/Qwen2.5-1.5B-Instruct",
#         "prompt": "San Francisco is a",
#         "max_tokens": 7,
#         "temperature": 0
#     }'

module load gcc/15.1.0
module load cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export TORCH_CUDA_ARCH_LIST="9.0" 

FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir


## cuda 12.8 torch 2.7


pip install torch==2.7.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 vllm==0.10.0  --no-cache-dir


# Test Torch
# python -c "import torch; print(torch.cuda.is_available())"

# Test vLLM
# vllm serve Qwen/Qwen2.5-1.5B-Instruct
# curl http://localhost:8000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "Qwen/Qwen2.5-1.5B-Instruct",
#         "prompt": "San Francisco is a",
#         "max_tokens": 7,
#         "temperature": 0
#     }'

module load gcc/13.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)

FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn \
    --no-build-isolation \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128


#### torch 2.9.0 cuda 12.8



conda create -n vseek-vllm python=3.11
conda activate vseek-vllm
conda install cuda -c nvidia/label/cuda-12.8.1
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)

pip install torch==2.9.0 torchvision torchaudio 'numpy<2' --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="90-real"
export CMAKE_CUDA_ARCHITECTURES="90-real"

pip install vllm==0.12.0 'numpy<2' --no-build-isolation --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128

# Test Torch
python -c "import torch; print(torch.cuda.is_available())"

# Install from PyPI (default index), but allow PyTorch index for dependencies if needed
module load gcc/14.2.0 cuda/12.8

export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="90-real"
export CMAKE_CUDA_ARCHITECTURES="90-real"


FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn \
    --no-build-isolation \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128

pip install -e .
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl


vllm serve Qwen/Qwen2.5-1.5B-Instruct
curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
         "model": "Qwen/Qwen2.5-1.5B-Instruct",
         "prompt": "San Francisco is a",
         "max_tokens": 7,
         "temperature": 0
     }'

----------------------------------------------
### Cuda 12.8 torch 2.7.1 vllm 0.11.0 custom build Most stable build

export TORCH_CUDA_ARCH_LIST="9.0" 
module load gcc/13.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)


conda create -n vseek-vllm python=3.11
conda activate vseek-vllm

conda install cuda -c nvidia/label/cuda-12.8.1
pip install torch==2.7.1 'numpy<2' torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   --no-cache-dir



cd $WORK
rm -rf vllm
git clone https://github.com/vllm-project/vllm.git
git checkout releases/v0.11.0

# ============================================
# PRE-BUILD CHECKS
# ============================================
echo "=== Pre-build environment check ==="
module load gcc/13.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')


export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="90-real"
export CMAKE_CUDA_ARCHITECTURES="90-real"

# Verify environment before building
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC (should be gcc 13.2)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
$CC --version | head -1

# Clean build (run this if rebuilding or if SM targets are wrong)
rm -rf .deps/ build/ vllm/*.so vllm/**/*.so

python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/common.txt
cat > /tmp/constraints.txt << 'EOF'
numpy<2
EOF

MAX_JOBS=16 pip install -e . --no-build-isolation --no-cache-dir --constraint /tmp/constraints.txt


# ============================================
# VERIFICATION CHECKS - Run after vLLM build
# ============================================

echo "=== 1. Environment Check ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
nvcc --version | grep "release"

echo ""
echo "=== 2. Python Package Versions ==="
python -c "
import torch
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch CUDA: {torch.version.cuda}')
print(torch.cuda.is_available())
print(f'NumPy: {np.__version__}')
assert 'cu128' in torch.__version__ or torch.version.cuda == '12.8', 'ERROR: PyTorch not built with CUDA 12.8!'
assert int(np.__version__.split('.')[0]) < 2, 'ERROR: NumPy >= 2.0 detected!'
print('✓ PyTorch CUDA 12.8 verified')
print('✓ NumPy < 2 verified')
"

echo ""
echo "=== 3. vLLM Flash Attention SM Targets ==="
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
echo "vLLM path: $VLLM_PATH"

check_sm() {
    local so_file="$1"
    if [ -f "$so_file" ]; then
        local targets=$(strings "$so_file" 2>/dev/null | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
        if echo "$targets" | grep -q "sm_90"; then
            echo "[PASS] $(basename $so_file): $targets"
        else
            echo "[FAIL] $(basename $so_file): $targets (MISSING sm_90)"
            return 1
        fi
    fi
}

check_sm "$VLLM_PATH/_C.abi3.so"
check_sm "$VLLM_PATH/vllm_flash_attn/_vllm_fa2_C.abi3.so"
check_sm "$VLLM_PATH/vllm_flash_attn/_vllm_fa3_C.abi3.so"

echo ""
echo "=== 4. GLIBCXX Check ==="
strings /lib64/libstdc++.so.6 2>/dev/null | grep GLIBCXX | tail -3
# Required: GLIBCXX_3.4.32 or higher for flash_attn

echo ""
echo "=== 5. GPU Check (run on compute node) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
else
    echo "(nvidia-smi not available - run this on compute node)"
fi

echo ""
echo "=== Build Verification Complete ==="

cd $WORK
rm -rf triton
git clone https://github.com/triton-lang/triton.git  
cd triton 
git checkout release/3.3.x  
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 pip install -e python --verbose

## Build VSEEK and VERL


cd $HOME/VSeek-R1

# Install from PyPI (default index), but allow PyTorch index for dependencies if needed


pip install -e .
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl

#####Sanity checks


# Check what CUDA arch flash_attn was built for
python -c "
import flash_attn
import os
fa_path = os.path.dirname(flash_attn.__file__)
print('flash_attn path:', fa_path)
" && find $(python -c "import flash_attn; import os; print(os.path.dirname(flash_attn.__file__))") -name "*.so" -exec sh -c 'echo "=== {} ===" && strings {} | grep -E "sm_[0-9]+" | head -3' \;

# Check vLLM's compiled extensions
python -c "
import vllm
import os
vllm_path = os.path.dirname(vllm.__file__)
print('vLLM path:', vllm_path)
" && find $(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))") -name "*.so" -exec sh -c 'echo "=== {} ===" && strings {} | grep -E "sm_[0-9]+" | head -3' \; 2>/dev/null | head -50

# Check GPU compute capability on compute node
nvidia-smi --query-gpu=compute_cap --format=csv

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv


module load gcc/13.2.0 cuda/12.8
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which gcc)
export CXX=$(which g++)

FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn \
    --no-build-isolation \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128


module load gcc/13.2.0 cuda/12.8
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which gcc)
export CXX=$(which g++)


<!-- pip install -e .
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl -->


-------------------------------------------------------

### cuda 12.8 torch 2.9.0 


export TORCH_CUDA_ARCH_LIST="9.0" 
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)


conda create -n vseek-vllm2 python=3.11
conda activate vseek-vllm2

pip install torch==2.9.0 'numpy<2' torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   --no-cache-dir


cd $WORK/vllm
y
git checkout releases/v0.12.0

# ============================================
# PRE-BUILD CHECKS
# ============================================
echo "=== Pre-build environment check ==="
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')


export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="sm_90"

# Verify environment before building
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC (should be gcc 13.2)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
$CC --version | head -1

# Clean build (run this if rebuilding or if SM targets are wrong)
rm -rf .deps/ build/ vllm/*.so vllm/**/*.so

python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/common.txt
cat > /tmp/constraints.txt << 'EOF'
numpy<2
EOF

MAX_JOBS=24 pip install -e . --no-build-isolation --no-cache-dir --constraint /tmp/constraints.txt

# ============================================
# VERIFICATION CHECKS - Run after vLLM build
# ============================================

echo "=== 1. Environment Check ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
nvcc --version | grep "release"

echo ""
echo "=== 2. Python Package Versions ==="
python -c "
import torch
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'NumPy: {np.__version__}')
assert 'cu128' in torch.__version__ or torch.version.cuda == '12.8', 'ERROR: PyTorch not built with CUDA 12.8!'
assert int(np.__version__.split('.')[0]) < 2, 'ERROR: NumPy >= 2.0 detected!'
print('✓ PyTorch CUDA 12.8 verified')
print('✓ NumPy < 2 verified')
"

echo ""
echo "=== 3. vLLM Flash Attention SM Targets ==="
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
echo "vLLM path: $VLLM_PATH"

check_sm() {
    local so_file="$1"
    if [ -f "$so_file" ]; then
        local targets=$(strings "$so_file" 2>/dev/null | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
        if echo "$targets" | grep -q "sm_90"; then
            echo "[PASS] $(basename $so_file): $targets"
        else
            echo "[FAIL] $(basename $so_file): $targets (MISSING sm_90)"
            return 1
        fi
    fi
}

check_sm "$VLLM_PATH/_C.abi3.so"
check_sm "$VLLM_PATH/vllm_flash_attn/_vllm_fa2_C.abi3.so"
check_sm "$VLLM_PATH/vllm_flash_attn/_vllm_fa3_C.abi3.so"

echo ""
echo "=== 4. GLIBCXX Check ==="
strings /lib64/libstdc++.so.6 2>/dev/null | grep GLIBCXX | tail -3
# Required: GLIBCXX_3.4.32 or higher for flash_attn

echo ""
echo "=== 5. GPU Check (run on compute node) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
else
    echo "(nvidia-smi not available - run this on compute node)"
fi

echo ""
echo "=== Build Verification Complete ==="

cd $HOME/VSeek-R1

# Install from PyPI (default index), but allow PyTorch index for dependencies if needed


pip install -e .
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl

#####Sanity checks


# Check what CUDA arch flash_attn was built for
python -c "
import flash_attn
import os
fa_path = os.path.dirname(flash_attn.__file__)
print('flash_attn path:', fa_path)
" && find $(python -c "import flash_attn; import os; print(os.path.dirname(flash_attn.__file__))") -name "*.so" -exec sh -c 'echo "=== {} ===" && strings {} | grep -E "sm_[0-9]+" | head -3' \;

# Check vLLM's compiled extensions
python -c "
import vllm
import os
vllm_path = os.path.dirname(vllm.__file__)
print('vLLM path:', vllm_path)
" && find $(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))") -name "*.so" -exec sh -c 'echo "=== {} ===" && strings {} | grep -E "sm_[0-9]+" | head -3' \; 2>/dev/null | head -50

# Check GPU compute capability on compute node
nvidia-smi --query-gpu=compute_cap --format=csv

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv


module load gcc/13.2.0 cuda/12.8
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which gcc)
export CXX=$(which g++)

FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn \
    --no-build-isolation \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128

pip install -e .
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl


--------------------------------------------------------
### #### torch cuda 12.6 gcc/13.2.0 

module load gcc/13.2.0 cuda/12.6
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)


conda create -n vseek-vllm4 python=3.11
conda activate vseek-vllm4

<!-- pip install torch torchvision torchaudio 'numpy<2' --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir -->

pip install setuptools-scm
MAX_JOBS=8 pip install vllm 'numpy<2' --no-build-isolation --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126

# Test Torch
# python -c "import torch; print(torch.cuda.is_available())"

# Install from PyPI (default index), but allow PyTorch index for dependencies if needed
module load gcc/13.2.0 cuda/12.6
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)

FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn \
    --no-build-isolation \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126

pip install -e .
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl




# Setup retriever server on vtacc
CUDA_VISIBLE_DEVICES=0 python3 src/vseek/tools/server.py retriever.window_size=8 dataset.name=['lvb','lvbench','videomme','mlvu'] retriever.index_path=/work/11123/harshgoel99/vista/vseek retriever.retrieval_model_path=/work/11123/harshgoel99/vista/model_weights/viclip/ViClip-InternVid-10M-FLT.pth retriever.text_encoder_model_path=/work/11123/harshgoel99/vista/model_weights/viclip/bpe_simple_vocab_16e6.txt.gz


export SLURM_JOB_NODELIST="c608-141,c608-142"
export SLURM_JOB_NUM_NODES=2



huggingface-cli download "Qwen/Qwen3-VL-4B-Thinking" --cache-dir "/work/11123/harshgoel99/vista/huggingface/hub" || python3 -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('Qwen/Qwen3-VL-4B-Thinking', trust_remote_code=True); AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-4B-Thinking', trust_remote_code=True)"


 export SLURM_JOB_NODELIST="c608-041,c608-042,c608-051,c608-052,c608-061,c608-062,c608-081,c608-082"

 export SLURM_JOB_NUM_NODES=8

export SLURM_JOB_NODELIST="c608-081,c608-082,c608-091,c608-092,c608-101,c608-102,c608-111,c608-112"

export SLURM_JOB_NUM_NODES=8


export SLURM_JOB_NODELIST="c608-042,c608-052,c608-061,c608-062,c608-102,c608-111,c608-112,c608-142"

export SLURM_JOB_NUM_NODES=8



export SLURM_JOB_NODELIST="c608-051,c608-052,c608-061,c608-062"
export SLURM_JOB_NUM_NODES=4



-------------------------------------------------------
-------------------------------------------------------------
## Torch 2.8 CUDA 12.9


export TORCH_CUDA_ARCH_LIST="9.0" 
module load gcc/15.1.0 cuda/12.9
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

conda create -n vseek-vllm2 python=3.11 git -y
conda activate vseek-vllm2
conda install -c nvidia cuda-toolkit=12.9 nccl -y

pip install --force-reinstall torch==2.8.0 'numpy<2' torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 

export TORCH_CUDA_ARCH_LIST="9.0" 
module load gcc/15.1.0 cuda/12.9
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -c "
import torch
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch CUDA: {torch.version.cuda}')
print(torch.cuda.is_available())
"

cd $WORK
rm -rf vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout releases/v0.11.0


echo "=== Pre-build environment check ==="
module load gcc/15.1.0 cuda/12.9
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')



export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="90-real"
export CMAKE_CUDA_ARCHITECTURES="90-real"

# Verify environment before building
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC (should be gcc 15.2)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
$CC --version | head -1

# Clean build (run this if rebuilding or if SM targets are wrong)
rm -rf .deps/ build/ vllm/*.so vllm/**/*.so

python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/common.txt
cat > /tmp/constraints.txt << 'EOF'
numpy<2
EOF

MAX_JOBS=16 pip install -e . --no-build-isolation --no-cache-dir --constraint /tmp/constraints.txt --verbose

# ============================================
# VERIFICATION CHECKS - Run after vLLM build
# ============================================

echo "=== 1. Environment Check ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
nvcc --version | grep "release"

echo ""
echo "=== 2. Python Package Versions ==="
python -c "
import torch
import numpy as np
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch CUDA: {torch.version.cuda}')
print(torch.cuda.is_available())
print(f'NumPy: {np.__version__}')
assert 'cu129' in torch.__version__ or torch.version.cuda == '12.9', 'ERROR: PyTorch not built with CUDA 12.9!'
assert int(np.__version__.split('.')[0]) < 2, 'ERROR: NumPy >= 2.0 detected!'
print('✓ PyTorch CUDA 12.9 verified')
print('✓ NumPy < 2 verified')
"

echo ""
echo "=== 3. vLLM Flash Attention SM Targets ==="
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
echo "vLLM path: $VLLM_PATH"

check_sm() {
    local so_file="$1"
    if [ -f "$so_file" ]; then
        local targets=$(strings "$so_file" 2>/dev/null | grep -oE "sm_[0-9]+" | sort -u | tr '\n' ' ')
        if echo "$targets" | grep -q "sm_90"; then
            echo "[PASS] $(basename $so_file): $targets"
        else
            echo "[FAIL] $(basename $so_file): $targets (MISSING sm_90)"
            return 1
        fi
    fi
}

check_sm "$VLLM_PATH/_C.abi3.so"
check_sm "$VLLM_PATH/vllm_flash_attn/_vllm_fa2_C.abi3.so"
check_sm "$VLLM_PATH/vllm_flash_attn/_vllm_fa3_C.abi3.so"

echo ""
echo "=== 4. GLIBCXX Check ==="
strings /lib64/libstdc++.so.6 2>/dev/null | grep GLIBCXX | tail -3
# Required: GLIBCXX_3.4.32 or higher for flash_attn

echo ""
echo "=== 5. GPU Check (run on compute node) ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
else
    echo "(nvidia-smi not available - run this on compute node)"
fi

echo ""
echo "=== Build Verification Complete ==="


#####Sanity checks


# Check what CUDA arch flash_attn was built for
python -c "
import flash_attn
import os
fa_path = os.path.dirname(flash_attn.__file__)
print('flash_attn path:', fa_path)
" && find $(python -c "import flash_attn; import os; print(os.path.dirname(flash_attn.__file__))") -name "*.so" -exec sh -c 'echo "=== {} ===" && strings {} | grep -E "sm_[0-9]+" | head -3' \;

# Check vLLM's compiled extensions
python -c "
import vllm
import os
vllm_path = os.path.dirname(vllm.__file__)
print('vLLM path:', vllm_path)
" && find $(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))") -name "*.so" -exec sh -c 'echo "=== {} ===" && strings {} | grep -E "sm_[0-9]+" | head -3' \; 2>/dev/null | head -50

# Check GPU compute capability on compute node
nvidia-smi --query-gpu=compute_cap --format=csv

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv

module load gcc/15.1.0 cuda/12.9
export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which gcc)
export CXX=$(which g++)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="90-real"
export CMAKE_CUDA_ARCHITECTURES="90-real"

cd $WORK
rm -rf triton
git clone https://github.com/triton-lang/triton.git  
cd triton 
git checkout release/3.4.x  
pip install -r python/requirements.txt
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 pip install -e . --verbose

## Build VSEEK and VERL


<!-- module load gcc/14.2.0 
export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="90-real"
export CMAKE_CUDA_ARCHITECTURES="90-real"

export CUDA_HOME="$CONDA_PREFIX"
export CC=$(which gcc)
export CXX=$(which g++)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=16 pip install flash-attn \
    --no-build-isolation \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --verbose -->

module load gcc/15.1.0 cuda/12.9
cd $HOME/VSeek-R1
pip install -e .
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl



####################################################

module load gcc/14.2.0 cuda/12.8
conda create --name vseek-vllm3 python=3.12
conda activate vseek-vllm3
module load gcc/14.2.0 cuda/12.8
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir


cd $WORK
rm -rf vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout releases/v0.12.0

# ============================================
# PRE-BUILD CHECKS
# ============================================
echo "=== Pre-build environment check ==="
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')


export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="sm_90"

# Verify environment before building
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC (should be gcc 13.2)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
$CC --version | head -1

# Clean build (run this if rebuilding or if SM targets are wrong)
rm -rf .deps/ build/ vllm/*.so vllm/**/*.so

python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/common.txt
sed -i 's/^flashinfer-python.*/flashinfer-python/' requirements/cuda.txt
grep flashinfer-python requirements/cuda.txt  # Verify
cat > /tmp/constraints.txt << 'EOF'
numpy<2
EOF

MAX_JOBS=24 pip install -e . --no-build-isolation --no-cache-dir --constraint /tmp/constraints.txt --verbose


pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard --no-cache-dir
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" --no-cache-dir
FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
pip install flashinfer-python==0.3.1 --no-cache-dir 
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"
pip install --no-deps -e .


#######################################################


module load gcc/14.2.0 cuda/12.8
conda create --name vseek-vllm4 python=3.12 -y
conda activate vseek-vllm4
module load gcc/14.2.0 cuda/12.8
pip install torch==2.10.0 torchvision torchaudio flashinfer-python --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir


cd $WORK
rm -rf vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout releases/v0.12.0

# ============================================
# PRE-BUILD CHECKS
# ============================================
echo "=== Pre-build environment check ==="
module load gcc/14.2.0 cuda/12.8
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)
export CMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')


export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_FA_CMAKE_GPU_ARCHES="sm_90"

# Verify environment before building
echo "CUDA_HOME=$CUDA_HOME"
echo "CC=$CC (should be gcc 13.2)"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
$CC --version | head -1

# Clean build (run this if rebuilding or if SM targets are wrong)
rm -rf .deps/ build/ vllm/*.so vllm/**/*.so

python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/common.txt
cat > /tmp/constraints.txt << 'EOF'
numpy<2
EOF

MAX_JOBS=24 pip install -e . --no-build-isolation --no-cache-dir --constraint /tmp/constraints.txt --verbose
pip install torch==2.10.0 torchvision torchaudio flashinfer-python --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard --no-cache-dir
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1" --no-cache-dir
FLASH_ATTN_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir --verbose
pip install flashinfer-python==0.5.3 --no-cache-dir 
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

cd $HOME/VSeek-R1
pip install -r requirements_vllm_slurm2.txt --no-deps
pip install vendor/verl
pip install -e .