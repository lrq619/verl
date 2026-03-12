#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flowgrpo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

DATASET_DIR=/data/oss_bucket_0/borui/data

IMG=$HF_HOME/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6
REWARD_IMG=$HF_HOME/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

export VERL_VLLM_RPC_TIMEOUT_S=600

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh \
  trainer.n_gpus_per_node=8 \
  trainer.logger='["console","wandb"]' \
  trainer.default_local_dir=$HF_HOME/checkpoints/flow_grpo/qwen_image_ocr \
  +trainer.skip_initial_update_weights=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
  actor_rollout_ref.model.path=$IMG \
  actor_rollout_ref.model.tokenizer_path=$IMG/tokenizer \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  ++actor_rollout_ref.rollout.free_cache_engine=True \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.gpu_memory_utilization=0.20 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=900 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout=1800 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config='{cudagraph_mode:NONE}' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.val_kwargs.num_inference_steps=50 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.04 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  reward.reward_model.enable=True \
  reward.reward_model.rollout.gpu_memory_utilization=0.1 \
  reward.reward_model.rollout.max_model_len=32768 \
  reward.reward_model.rollout.max_num_seqs=4 \
  reward.reward_model.rollout.max_num_batched_tokens=2048 \
  reward.reward_model.rollout.tensor_model_parallel_size=4 \
  reward.reward_model.model_path=$REWARD_IMG \
  data.train_batch_size=32 \
  data.train_files=$DATASET_DIR/ocr/train.parquet \
  data.val_files=$DATASET_DIR/ocr/test.parquet \
  trainer.total_epochs=15 \
  hydra.run.dir=$LOG_DIR \
  hydra.output_subdir=null \
  +ray_kwargs.ray_init._temp_dir=/tmp/ray_tmp \
