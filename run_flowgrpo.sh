#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=./logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flowgrpo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

IMG=/hf_home/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6
REWARD_IMG=/hf_home/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

export VERL_VLLM_RPC_TIMEOUT_S=600

CUDA_VISIBLE_DEVICES=0,1 bash examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh \
  trainer.n_gpus_per_node=2 \
  trainer.logger='["console"]' \
  trainer.default_local_dir=/hf_home/checkpoints/flow_grpo/qwen_image_ocr \
  +trainer.skip_initial_update_weights=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
  actor_rollout_ref.model.path=$IMG \
  actor_rollout_ref.model.tokenizer_path=$IMG/tokenizer \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  ++actor_rollout_ref.rollout.free_cache_engine=True \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.gpu_memory_utilization=0.20 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=900 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout=1800 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config='{cudagraph_mode:NONE}' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  reward.reward_model.enable=True \
  reward.reward_model.rollout.gpu_memory_utilization=0.1 \
  reward.reward_model.rollout.max_model_len=32768 \
  reward.reward_model.rollout.max_num_seqs=4 \
  reward.reward_model.rollout.max_num_batched_tokens=2048 \
  reward.reward_model.rollout.tensor_model_parallel_size=2 \
  reward.reward_model.model_path=$REWARD_IMG \
  data.train_max_samples=4 \
  data.train_batch_size=4 \
  data.val_max_samples=16 \
  data.train_files=/workspace/data/ocr/train.parquet \
  data.val_files=/workspace/data/ocr/test.parquet \
  hydra.run.dir=$LOG_DIR \
  hydra.output_subdir=null \
  +ray_kwargs.ray_init._temp_dir=/workspace/ray_tmp \
