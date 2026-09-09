#!/usr/bin/env bash
set -euo pipefail

# Experimental one-node smoke wrapper for the gated 67B vision checkpoint.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained}"
MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-75000}"
TRUNCATE_NUM_LAYERS="${TRUNCATE_NUM_LAYERS:-16}" # Full model: 89 layers; leave empty to use all.

GPU_MEMORY_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | awk 'NR == 1 { print $1 }')"
if [[ -z "${GPU_MEMORY_MIB}" || "${GPU_MEMORY_MIB}" -lt "${MIN_GPU_MEMORY_MIB}" ]]; then
  echo "The 1-node 67B layout requires GPUs with at least ${MIN_GPU_MEMORY_MIB} MiB each; found ${GPU_MEMORY_MIB:-unknown}." >&2
  echo "Use the 8-node submit wrapper on 80 GB GPUs." >&2
  exit 1
fi

export MODEL_NAME
export MAX_STEPS="${MAX_STEPS:-5}"
# Long microbatch sequence length on 1N with no CP.
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-2048}"
# Step Tokens <= P * G * min(PROMPT_LEN + MAX_NEW_TOKENS, MAX_SEQUENCE_LENGTH)
export NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-2}"
export NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
# CLEVR max tokens per generation, we probably don't need more.
# These tokens are all trained on, i.e. max valid tokens = P * G * MAX_NEW_TOKENS.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# To do small-scale measurements to estimate max sequence length and CP.
export IGNORE_EOS="${IGNORE_EOS:-true}"
# Number of training samples per global step.
export TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
# Grad Accum ~ Step Tokens / (POLICY_DP * MAX_SEQUENCE_LENGTH * TRAIN_MICRO_BATCH_SIZE)
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
# Maximum packed training tokens per microbatch.
export TRAIN_MB_TOKENS="${TRAIN_MB_TOKENS:-$((TRAIN_MICRO_BATCH_SIZE * MAX_SEQUENCE_LENGTH))}"
export LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-${TRAIN_MICRO_BATCH_SIZE}}"
# No precision-aware optimizer state. Use offloading esp. when computing logprobs.
export USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-false}"
export PREFIX_CACHING_MAMBA_GB="${PREFIX_CACHING_MAMBA_GB:-20}"
export ENABLE_NSYS="${ENABLE_NSYS:-false}"
export MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL="${MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL:-100}"
if [[ "${ENABLE_NSYS}" == "true" ]]; then
  export NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-2:5}"
fi

TRUNCATION_OVERRIDES=()
if [[ -n "${TRUNCATE_NUM_LAYERS}" ]]; then
  TRUNCATION_OVERRIDES=(
    "++policy.megatron_cfg.truncate_num_layers=${TRUNCATE_NUM_LAYERS}"
    "++policy.generation.vllm_kwargs.nemo_truncate_num_layers=${TRUNCATE_NUM_LAYERS}"
  )
fi

# Nemotron Super Omni Benchmarking (1-Node Full Model)
exec bash "${SCRIPT_DIR}/run_nemotron_omni_multimodal_single_controller_1n4g.sh" \
  "${TRUNCATION_OVERRIDES[@]}" \
  policy.sequence_packing.train_mb_tokens="${TRAIN_MB_TOKENS}" \
  policy.logprob_batch_size="${LOGPROB_BATCH_SIZE}" \
  policy.megatron_cfg.optimizer.optimizer_cpu_offload=true \
  policy.megatron_cfg.optimizer.optimizer_offload_fraction=1.0 \
  policy.offload_optimizer_for_logprob=true \
  "$@"
