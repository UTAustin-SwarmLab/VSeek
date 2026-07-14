# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535
export MASTER_PORT=${MASTER_PORT:-29651}
export CUDA_LAUNCH_BLOCKING=1
  # rerun your training
PROJECT_DIR="../../../src/vseek/"
CONFIG_PATH="$PROJECT_DIR/config"
MODEL_PATH="Qwen/Qwen3-VL-4B-Thinking"
python3 -m vseek.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='lvb_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=2192 \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=98304 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.free_cache_engine=True  \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.format="hermes" \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="vseek" \
    trainer.experiment_name="test" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    data.train_files=/work/11123/harshgoel99/vista/vseek/tagsummary/train.parquet \
    data.val_files=/work/11123/harshgoel99/vista/vseek/tagsummary/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config.tools.0.config.retriever.server_url="http://${HEAD_NODE}:9000" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="src/vseek/config/retriever/toolconfig.yaml" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="src/vseek/config/retriever/vllm_agent.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop="vseek_tag_summary_agent" \
    trainer.total_epochs=50 \
    trainer.val_before_train=True \
    trainer.log_val_generations=20

