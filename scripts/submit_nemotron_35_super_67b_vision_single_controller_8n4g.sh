#!/usr/bin/env bash
set -euo pipefail

# Eight-node wrapper for the gated Nemotron 3.5 Super 67B vision checkpoint.
# Defaults to a non-colocated 16-GPU training / 16-GPU generation split.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TASK="${TASK:-clevr}"
export MODEL_NAME="${MODEL_NAME:-nvidia/NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained}"
export JOB_NAME="${JOB_NAME:-nemotron-35-super-67b-vision-single-controller-8n4g}"
export RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/../workspace/results/nemo-rl-v2-super-67b-vision/${TASK}-8n4g}"
# Grace-Blackwell Benchmarking
export NUM_NODES="${NUM_NODES:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export MAX_STEPS="${MAX_STEPS:-25}"
# Tokens per request / rollout, i.e. prompt + generation tokens.
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-49152}"
# Per-step chunked-prefill token budget. 
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-8192}"
# Max generation tokens.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
# Prompts and Generations
export NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"
export NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
export TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
export LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-${TRAIN_MICRO_BATCH_SIZE}}"
# STEP_TOKENS <= P * G * min(PROMPT_LEN + MAX_NEW_TOKENS, MAX_SEQUENCE_LENGTH)
# TRAIN_MB_TOKENS and LOGPROB_MB_TOKENS should divide up STEP_TOKENS.
export TRAIN_MB_TOKENS="${TRAIN_MB_TOKENS:-$((TRAIN_MICRO_BATCH_SIZE * MAX_SEQUENCE_LENGTH))}"
export LOGPROB_MB_TOKENS="${LOGPROB_MB_TOKENS:-$((LOGPROB_BATCH_SIZE * MAX_SEQUENCE_LENGTH * 1))}"
# Training Parallelism
export POLICY_TP="${POLICY_TP:-8}"
export POLICY_EP="${POLICY_EP:-16}"
# CP is not very useful for CLEVR - prompt + gen token count is small.
# If the task has a larger prompt and generation requirement, then CP is needed.
# However, CP is preferable to multiple sequence-packed microbatches.
export POLICY_CP="${POLICY_CP:-2}"
# Generation Parallelism
export NUM_GEN_NODES="${NUM_GEN_NODES:-4}"
export INFER_TP="${INFER_TP:-8}"
export INFER_EP="${INFER_EP:-8}"
# GPU-resident attention KV blocks and reusable hybrid-Mamba prefix state.
export BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-16}"
export PREFIX_CACHING_MAMBA_GB="${PREFIX_CACHING_MAMBA_GB:-20}"
# Keep embeddings for the roughly NUM_PROMPTS_PER_STEP distinct CLEVR images resident.
export VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-2147483648}"
# Optimizer state CPU full offloading, log-prob offloading, or not offloaded.
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
export OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-true}"
export OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-true}"
# Skip the gradient buffer deallocation and reallocation surrounding refit and logprob.
export OFFLOAD_POLICY_BEFORE_REFIT="${OFFLOAD_POLICY_BEFORE_REFIT:-false}"
# Disables some torch.cuda.empty_cache() calls in the RL loop.
export EMPTY_UNUSED_MEMORY_LEVEL="${EMPTY_UNUSED_MEMORY_LEVEL:-0}"
# No BF16 or FP8 optimizer state allowed for actual NT training.
export USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-false}"
NUM_TRAIN_NODES=$((NUM_NODES - NUM_GEN_NODES))
TRAIN_DP_SIZE=$((NUM_TRAIN_NODES * GPUS_PER_NODE / (POLICY_TP * POLICY_CP)))
# CUDA Caching Allocator Configs - Only use if needed, not performant in RL.
# expandable_segments:True will recover a lot of reserved memory but nukes alloc perf in RL.
# particularly for Megatron training due to DDP gradient buffer de-allocation and re-allocation.
# max_split_size_mb:X will prevent splitting blocks larger than X megabytes to avoid large
# re-allocations (like DDP buffers) while still servicing smaller allocations via splitting
# instead of reserving more memory.
# backend:cudaMallocAsync will asynchronously manage the memory pool to avoid fragmented
# reserved memory while improving allocation performance.
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Experiment Params
export ENABLE_NSYS="${ENABLE_NSYS:-true}"
export MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL="${MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL:-100}"
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_NAME="${WANDB_NAME:-nt-omni-super-67b-${TASK}-${NUM_NODES}n${GPUS_PER_NODE}g-tr${NUM_TRAIN_NODES}n-gen${NUM_GEN_NODES}n-tp${POLICY_TP}dp${TRAIN_DP_SIZE}ep${POLICY_EP}cp${POLICY_CP}-itp${INFER_TP}iep${INFER_EP}-seq${MAX_SEQUENCE_LENGTH}-trainmbtok${TRAIN_MB_TOKENS}lpmbtok${LOGPROB_MB_TOKENS}-p${NUM_PROMPTS_PER_STEP}g${NUM_GENERATIONS_PER_PROMPT}-gbs${TRAIN_GBS}mbs${TRAIN_MICRO_BATCH_SIZE}lpbs${LOGPROB_BATCH_SIZE}-cpuoff${OPTIMIZER_CPU_OFFLOAD}}"
export EXTRA_OVERRIDES="++policy.sequence_packing.train_mb_tokens=${TRAIN_MB_TOKENS} ++policy.sequence_packing.logprob_mb_tokens=${LOGPROB_MB_TOKENS} ++policy.logprob_batch_size=${LOGPROB_BATCH_SIZE} ++policy.generation.mcore_generation_config.offload_policy_before_refit=${OFFLOAD_POLICY_BEFORE_REFIT} ++policy.megatron_cfg.empty_unused_memory_level=${EMPTY_UNUSED_MEMORY_LEVEL} ${EXTRA_OVERRIDES:-}"

exec bash "${SCRIPT_DIR}/submit_nemotron_omni_multimodal_single_controller_8n4g.sh" "$@"
