Next, install uv:
```bash
pip install uv
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

export CUDA_HOME=...
pip install -r requirements_sglang.txt
pip install flash-attn==2.8.3 --no-build-isolation
export CUDA_HOME="$CONDA_PREFIX"
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
bash scripts/preprocess_mvlu.sh tagsummary
```

### CGBench


#### Unzip Videos and convert parquet
```bash
SUBTITLE_DIR=/nas/mars/dataset/CGBench CHUNKS_DIR=/nas/mars/dataset/CGBench OUTPUT_DIR=/nas/mars/dataset/CGBench/videos bash scripts/utils/unzip_videos.sh
```

#### Burn Subtitles

```bash
python3 scripts/utils/burn_subtitles.py --json-file cgbench_mini.json --output-dir /nas/mars/dataset/CGBench/burn-subtitles --data-folder /nas/mars/dataset/CGBench


### Run indexing pipeline
```bash
    python3 scripts/data_ops/run_data_pipeline.py \
    dataset.name=cgbench \
    retriever.window_size=8 \
    dataset.videomme.dataset_path=/nas/mars/dataset/CGBench/ \
    dataset.videomme.burned_path=/nas/mars/dataset/CGBench/burn-subtitles/ \
    retriever.index_path=/home/hg22723/vseek/dataset \
    retriever.gpu_number=7
```


```bash
bash scripts/data_ops/preprocess_cgbench.sh \
    --local_dataset_path "/nas/mars/dataset/CGBench" \
    --burned_path "/nas/mars/dataset/CGBench/burn-subtitles" \
    --train_ratio 0.8 \
    --local_save_dir /nas/mars/dataset/CGBench \
    --index_path /home/hg22723/vseek/dataset \
    --window_size 8 \
    --prompt_type tagsummary
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


### TACC build

CC=/usr/bin/gcc CXX=/usr/bin/g++ pip install -r requirements.txt

install decord from source 
- install ffmpeg=4.4.2 with conda install 'ffmpeg=4'

cd to decord

mdkir build & cd build
C=$(which gcc) CXX=$(which g++) cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=$WORK/miniforge3/envs/vseek-vllm
make

cd python
python setup.py install



### Puls related fix

python "/home/hg22723/projects/VSeek-R1/src/vseek/puls/test/evaluate_puls_with_gpt.py" --input_json "/nas/mars/dataset/MLVU/MLVU/puls.json" --output_jsonl "/home/hg22723/projects/VSeek-R1/tmp/mlvu_puls_eval_smoke.jsonl" --summary_json "/home/hg22723/projects/VSeek-R1/tmp/mlvu_puls_eval_smoke_summary.json" --model gpt-4o --max_samples 200 --seed 42