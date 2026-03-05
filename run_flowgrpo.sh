#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=/workspace/verl/logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/flowgrpo_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

IMG=/hf_cache/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6
REWARD_IMG=/hf_cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

export VERL_VLLM_RPC_TIMEOUT_S=600

CUDA_VISIBLE_DEVICES=0,1 bash examples/flowgrpo_trainer/run_flowgrpo_nocfg.sh \
  trainer.n_gpus_per_node=2 \
  trainer.logger='["console"]' \
  +trainer.skip_initial_update_weights=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.model.path=$IMG \
  actor_rollout_ref.model.tokenizer_path=$IMG/tokenizer \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=0 \
  ++actor_rollout_ref.rollout.free_cache_engine=True \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.gpu_memory_utilization=0.20 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=900 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.init_timeout=1800 \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.custom_pipeline=verl.utils.vllm_omni.pipelines.QwenImagePipelineWithLogProb \
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config='{cudagraph_mode:NONE}' \
  reward.reward_model.enable=False \
  reward.reward_model.model_path=$REWARD_IMG \
  data.train_files=/root/data/ocr/train.parquet \
  data.val_files=/root/data/ocr/test.parquet \
  hydra.run.dir=/workspace/verl/logs \
  hydra.output_subdir=null \
