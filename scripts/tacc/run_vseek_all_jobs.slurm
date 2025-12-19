#!/bin/bash
#SBATCH -J vseek_job
#SBATCH -o vseek_job-%j.out
#SBATCH -e vseek_job-%j.err
#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --cpus-per-task=64
#SBATCH -t 48:00:00
#SBATCH --mem=0

# USAGE: sbatch run_vseek_job.slurm

echo "Starting job on $(hostname)"
date

# Activate conda environment on head node
source ~/.bashrc
conda activate vseek-vllm
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set HuggingFace cache to $WORK directory (persists across jobs, more space)
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

# -------------------------------------------------------------------------
# Ray Multi-Node Setup & Head Node Detection
# -------------------------------------------------------------------------
# Getting the node names
# Filter out TACC banners - only keep lines that start with alphanumeric (valid hostnames)
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -E '^[a-zA-Z0-9]')
nodes_array=($nodes)
HEAD_NODE=${nodes_array[0]}

if [ -z "$HEAD_NODE" ]; then
    echo "Error: Could not determine HEAD_NODE from SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "Raw scontrol output:"
    scontrol show hostnames "$SLURM_JOB_NODELIST" | cat -A
    exit 1
fi

echo "Head Node: $HEAD_NODE"

# Get IP of head node
# TACC systems print banners that can pollute command output
# We try multiple methods to reliably get the IP

# Method 1: Use getent hosts (DNS/hosts lookup, no srun needed)
head_node_ip=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -n 1)

# Method 2: If getent failed, try Python socket (very reliable)
if [ -z "$head_node_ip" ]; then
    echo "getent failed, trying Python socket..."
    head_node_ip=$(python3 -c "import socket; print(socket.gethostbyname('$HEAD_NODE'))" 2>/dev/null)
fi

# Method 3: srun hostname -I with grep (original approach)
if [ -z "$head_node_ip" ]; then
    echo "Python failed, trying srun hostname -I..."
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I 2>/dev/null | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -n 1)
fi

if [ -z "$head_node_ip" ]; then
    echo "Error: Could not determine head node IP by any method."
    exit 1
fi

echo "Resolved head node IP: $head_node_ip"

# Port for Ray
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Ray Head IP: $ip_head"

# Clean up any stale Ray processes/sessions from previous runs
echo "Cleaning up stale Ray sessions..."
# Use srun to cleanup on all nodes
srun scripts/tacc/ray_wrapper.sh bash -c "ray stop --force; rm -rf /tmp/ray/session_*" || true

echo "Starting Ray Head on $HEAD_NODE"
export RAY_ADDRESS=$ip_head

# -------------------------------------------------------------------------
# Launch Ray Cluster using srun and wrapper
# -------------------------------------------------------------------------

# 1. Start Ray Head on the Head Node
# --block is required to keep srun alive, but we background srun itself
# num-gpus=0 to reserve GPU for Server
echo "Launching Ray Head..."
srun --nodes=1 --ntasks=1 --w "$HEAD_NODE" \
    scripts/tacc/ray_wrapper.sh \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus=1 --dashboard-host=0.0.0.0 --block &
sleep 10

# 2. Start Ray Workers on remaining nodes
# num-gpus=1 for training
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ $worker_num -gt 0 ]; then
    echo "Launching Ray Workers on $worker_num nodes..."
    srun --nodes=$worker_num --ntasks=$worker_num --exclude="$HEAD_NODE" \
        scripts/tacc/ray_wrapper.sh \
        ray start --address "$ip_head" --num-gpus=1 --block &
    sleep 10
fi

# Wait for workers to register and verify cluster
echo "Ray cluster status:"
# Run status check inside wrapper to ensure environment is right
scripts/tacc/ray_wrapper.sh ray status || echo "Ray status check failed (might be too early)"

# Verify we have enough GPUs
total_gpus=$(scripts/tacc/ray_wrapper.sh ray status | grep -oP '\d+\.\d+(?=/\d+\.\d+ GPU)' | tail -1 || echo "0")
echo "Total GPUs in cluster: $total_gpus"

# -------------------------------------------------------------------------

# Start the server in the background on the Head Node
# We bind it to GPU 0 as requested
echo "Starting server..."
# Use wrapper to ensure env is correct, but run locally as we are on Head Node
# scripts/tacc/ray_wrapper.sh python3 -u src/vseek/tools/server.py \
#     retriever.window_size=8 \
#     "dataset.name=['lvb','lvbench','videomme','mlvu']" \
#     retriever.server_url="http://${head_node_ip}:9000" \
#     > server_log.txt 2>&1 &
# SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for server to initialize..."
sleep 60  # Give the server time to load models and indices

# Run the training job
echo "Starting training..."

# IMPORTANT: The training job typically needs access to GPUs. 
# We unset CUDA_VISIBLE_DEVICES so the training script can see all available GPUs (including GPU 0).
# Note: If the server is using GPU 0 heavily, this might cause contention.


# Setup training variables
# Use absolute paths to avoid Hydra/OmegaConf relative path issues
SCRIPT_DIR="/home/11123/harshgoel99"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR="${PROJECT_ROOT}/src/vseek"
CONFIG_PATH="${PROJECT_DIR}/config"
MODEL_PATH="Qwen/Qwen3-VL-4B-Thinking"
PROJECT_NAME="vseek"
EXPERIMENT_NAME="qwen3-4bt_vl_lvbvmmelvbench-emreward-vllm-wtool-tagsummary"
CHECKPOINT_DIR="/work/11123/harshgoel99/vista/checkpoints"

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "CONFIG_PATH: $CONFIG_PATH"

# Training command based on scripts/train/vseek_grpo_tag_qvl34Bt_vllm_tagsummary.sh
# Ensure we are in the project root
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: PROJECT_DIR $PROJECT_DIR not found."
    kill $SERVER_PID
    exit 1
fi

# Create a temporary tool config with the correct server URL
# This is necessary because the rollout worker reads the config file directly from disk
TOOL_CONFIG_SRC="${PROJECT_DIR}/config/retriever/toolconfig.yaml"
TOOL_CONFIG_TMP="toolconfig_${SLURM_JOB_ID}.yaml"

if [ ! -f "$TOOL_CONFIG_SRC" ]; then
    echo "Error: $TOOL_CONFIG_SRC not found. Ensure you are running from project root."
    kill $SERVER_PID
    exit 1
fi

# Replace localhost with HEAD_NODE in the config
sed "s/127.0.0.1/${HEAD_NODE}/g" "$TOOL_CONFIG_SRC" > "$TOOL_CONFIG_TMP"

set -x
ulimit -n 65535
export MASTER_PORT=${MASTER_PORT:-29651}
export CUDA_LAUNCH_BLOCKING=1

module load gcc/14.2.0
module load cuda/12.8
CUDA_LAUNCH_BLOCKING=1 python3 -m vseek.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='lvb_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=2536 \
    data.max_response_length=6000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=50 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=98304 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.max_model_len=8536 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.free_cache_engine=True  \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.format="hermes" \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=$((SLURM_JOB_NUM_NODES)) \
    +ray_kwargs.ray_init.address="auto" \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    data.train_files=/work/11123/harshgoel99/vista/vseek/tagsummary/train.parquet \
    data.val_files=/work/11123/harshgoel99/vista/vseek/tagsummary/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_TMP" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="src/vseek/config/retriever/vllm_agent.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop="vseek_tag_summary_agent" \
    trainer.total_epochs=50 \
    trainer.val_before_train=True \
    trainer.log_val_generations=20

# Cleanup server and temp files
echo "Training finished. Stopping server..."
kill $SERVER_PID
rm "$TOOL_CONFIG_TMP"

