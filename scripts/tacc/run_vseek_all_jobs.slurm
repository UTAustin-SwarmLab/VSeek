#!/bin/bash
#SBATCH -J vseek_job
#SBATCH -o vseek_job-%j.out
#SBATCH -e vseek_job-%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1    # Important for Ray: 1 task (daemon) per node
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
#SBATCH --mem=0

# USAGE: sbatch run_vseek_job.slurm

echo "Starting job on $(hostname)"
date

# ==========================================
# 1. CONFIGURATION & PORTS
# ==========================================
# You can change these ports explicitly here
export MASTER_PORT=${MASTER_PORT:-29651}
export CUDA_LAUNCH_BLOCKING=1
export VLLM_USE_V1=1
export RAY_memory_usage_threshold=0.98

export RAY_PORT=6379
export RAY_DASHBOARD_PORT=8265
export SERVER_PORT=9000
export RAY_MIN_WORKER_PORT=10002
export RAY_MAX_WORKER_PORT=19999

# Environment Setup
source ~/.bashrc
module load gcc/14.2.0 cuda/12.8
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME="$TACC_CUDA_DIR"
export PATH="${CUDA_HOME}/bin:${PATH}"

conda activate vseek-vllm4
echo "Conda environment: $CONDA_DEFAULT_ENV"

# HuggingFace Cache
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

# Setup Paths
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home1/11123/harshgoel99/VSeek-R1}"
PROJECT_DIR="${PROJECT_ROOT}/src/vseek"
CONFIG_PATH="${PROJECT_DIR}/config"
MODEL_PATH="Qwen/Qwen3-VL-4B-Thinking"
PROJECT_NAME="vseek"
EXPERIMENT_NAME="qwen3-4bt_vl_lvbvmmemlvu-pulsreward-vllm-wtool-tagsummary"
CHECKPOINT_DIR="/work/11123/harshgoel99/vista/checkpoints"
REWARD_TYPE="puls" # puls or em
PULS_THRESHOLD=0.14

WRAPPER_SCRIPT="${PROJECT_ROOT}/scripts/tacc/ray_wrapper.sh"
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Wrapper script not found at $WRAPPER_SCRIPT"
    exit 1
fi
chmod +x "$WRAPPER_SCRIPT"

# ==========================================
# 2. HEAD NODE IDENTIFICATION
# ==========================================
# Get the first node from the nodelist to be the head
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
HEAD_NODE=${nodes_array[0]}

if [ -z "$HEAD_NODE" ]; then
    echo "Error: Could not determine HEAD_NODE from SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
    echo "Raw scontrol output:"
    scontrol show hostnames "$SLURM_JOB_NODELIST" | cat -A
    exit 1
fi

echo "Head Node: $HEAD_NODE"

# Get IP address (robust method)
# On TACC/HPC, hostname -I often returns multiple IPs (InfiniBand, Ethernet, Docker)
# We usually want the one that starts with 10. (internal) or 192. or 172.
HEAD_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')

if [ -z "$HEAD_IP" ]; then
    HEAD_IP=$(python3 -c "import socket; print(socket.gethostbyname('$HEAD_NODE'))")
    
fi

echo "Head Node: $HEAD_NODE"
echo "Head IP:   $HEAD_IP"

# Export for Ray
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

# Clean up previous Ray sessions locally
echo "Cleaning up stale Ray sessions..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true

# ==========================================
# 3. LAUNCH RAY CLUSTER
# ==========================================

echo "----------------------------------------"
echo "Launching Ray Cluster"
echo "----------------------------------------"

# A. Start Ray Head (Local execution on the first node)
# We do not use srun here because the batch script is already running on the head node.
echo "Starting Ray Head on $HEAD_NODE..."
"$WRAPPER_SCRIPT" ray start --head \
    --node-ip-address="$HEAD_IP" \
    --port=$RAY_PORT \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --min-worker-port=$RAY_MIN_WORKER_PORT \
    --max-worker-port=$RAY_MAX_WORKER_PORT \
    --num-gpus=1 \
    --block &

# Store PID to kill later if needed, though 'ray stop' handles it
HEAD_PID=$!
sleep 10

# B. Start Ray Workers (Remote execution via srun)
# --relative=1 means "start from the 2nd node assigned to this job"
# This prevents srun from trying to start a worker on the Head Node
NUM_WORKERS=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= NUM_WORKERS; i++)); do
    node_i=${nodes_array[i]}
    echo "Starting Ray Worker on $node_i"
    # We use srun to execute the start command on worker nodes.
    # Note: We explicitly source bashrc/conda again to ensure the non-interactive srun shell has the env.
    srun --nodes=1 --ntasks=1 -w $node_i \
        bash -c "source ~/.bashrc && conda activate vseek-vllm4 && \
                $WRAPPER_SCRIPT ray start --address $HEAD_IP:$RAY_PORT --num-gpus=1 --num-cpus=8 --block" &
    sleep 5
done

# C. Wait for Cluster Initialization
echo "Waiting for Ray cluster to initialize..."
for i in {1..30}; do
    # Check if we have the expected number of nodes
    alive_nodes=$(ray status --address "$HEAD_IP:$RAY_PORT" 2>/dev/null | grep "node_" | wc -l)
    if [ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]; then
        echo "Ray Cluster Ready! ($alive_nodes nodes connected)"
        break
    fi
    sleep 5
done

# Print Status
ray status --address "$HEAD_IP:$RAY_PORT"




cd "$PROJECT_ROOT" || exit 1
echo "Working directory: $(pwd) PROJECT_ROOT: $PROJECT_ROOT"

# Handle Tool Config (Server URL Injection)
TOOL_CONFIG_SRC="${PROJECT_DIR}/config/retriever/toolconfig.yaml"
TOOL_CONFIG_TMP="toolconfig_${SLURM_JOB_ID}.yaml"

echo "TOOL_CONFIG_SRC: $TOOL_CONFIG_SRC"

# ==========================================
# 4. START TOOL SERVER (Optional)
# ==========================================
# Start the server in the background on the Head Node
# We bind it to GPU 0 as requested
echo "Starting server..."
SERVER_PORT=8901
SERVER_URL="http://0.0.0.0:${SERVER_PORT}"
# SERVER_URL="http://${HEAD_NODE}:${SERVER_PORT}"
# Use wrapper to ensure env is correct, but run locally as we are on Head Node
python3 -u ${PROJECT_ROOT}/src/vseek/tools/server.py \
    retriever.window_size=8 \
    "dataset.name=['lvb','lvbench','videomme','mlvu']" \
    retriever.server_url="$SERVER_URL" \
    > ${PROJECT_ROOT}/server_log_${SLURM_JOB_ID}.txt 2>&1 &
SERVER_PID=$!



echo "Server PID: $SERVER_PID"
echo "Waiting for server to initialize..."
sleep 120  # Give the server time to load models and indices


echo "Testing server..."
python3 -c "
import requests                                                                                                            
url = 'http://${HEAD_IP}:${SERVER_PORT}/search'
payload = {
    'query': 'test',
    'dataset_name': 'videomme',  
    'video_id': 'vzfTpidE5wg',     
    'topk': 5
}
try:
    resp = requests.get(url, params=payload)
    print(resp.status_code)
    print(resp.json())
except Exception as e:
    print(e)
"

python3 -c " 
import requests                                                                                                           
url = 'http://${HEAD_IP}:${SERVER_PORT}/search_subtitle'
payload = {
    'query': 'test',
    'dataset_name': 'videomme',  
    'video_id': 'vzfTpidE5wg',     
    'topk': 5
}
try:
    resp = requests.get(url, params=payload)
    print(resp.status_code)
    print(resp.json())
except Exception as e:
    print(e)
"

# Run the training job
echo "Starting training..."


# ==========================================
# 5. TRAINING JOB
# ==========================================


if [ -f "$TOOL_CONFIG_SRC" ]; then
    # Inject the specific Head IP and Server Port
    sed "s/127.0.0.1/${HEAD_IP}/g" "$TOOL_CONFIG_SRC" | sed "s/port: 9000/port: ${SERVER_PORT}/g" > "$TOOL_CONFIG_TMP"
else 
    echo "Warning: Tool config not found at $TOOL_CONFIG_SRC"
    touch "$TOOL_CONFIG_TMP" # Prevent crash if file missing
fi


# Training Variables

export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback and docker, use everything else
export NCCL_IB_DISABLE=0              # Enable InfiniBand/RDMA
export NCCL_NET_GDR_LEVEL=5           # Enable GPU Direct RDMA
export NCCL_DEBUG=WARN       


echo "Starting Training..."
module load gcc/14.2.0 cuda/12.8
# Note: We do NOT use srun here. The python script runs on the head node
# and connects to the Ray cluster (address="auto") to distribute work.
python3 -m vseek.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='lvb_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=2536 \
    data.max_response_length=5656 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=50 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.actor.entropy_coeff=0.0001 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    +ray_kwargs.ray_init.address="auto" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.free_cache_engine=True  \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.format="hermes" \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=$((SLURM_JOB_NUM_NODES)) \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.max_actor_ckpt_to_keep=1 \
    data.train_files=/work/11123/harshgoel99/vista/vseek/tagsummary/train.parquet \
    data.val_files=/work/11123/harshgoel99/vista/vseek/tagsummary/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_TMP" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/src/vseek/config/retriever/vllm_agent.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop="vseek_tag_summary_agent" \
    trainer.total_epochs=50 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    custom_reward_function.path="${PROJECT_ROOT}/src/vseek/trainer/reward/vseekreward.py" \
    custom_reward_function.reward_kwargs.reward_type=$REWARD_TYPE \
    custom_reward_function.reward_kwargs.puls_threshold=$PULS_THRESHOLD \


# ==========================================
# 6. CLEANUP
# ==========================================
echo "Job finished. Cleaning up..."

# Kill Server if running
if [ ! -z "$SERVER_PID" ]; then
    kill $SERVER_PID
fi

ray stop --force