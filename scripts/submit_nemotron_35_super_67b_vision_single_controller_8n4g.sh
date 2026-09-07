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
# Data Config for CLEVR (Modify for VSTAT)
export MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-49152}"
export INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-4096}" # Per-step chunked-prefill token budget.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-8}"
export NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
export TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
export TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
# Training Parallelism (4n4g DP1 TP8 CP2 EP16)
# Needed to support the large model size and sequence length.
export POLICY_TP="${POLICY_TP:-8}"
export POLICY_EP="${POLICY_EP:-16}"
export POLICY_CP="${POLICY_CP:-2}"
# Generation Parallelism (4n4g DP2 TP8 EP8)
# More DP for generation speedup and less model parallelism
# due to fewer activations, gradients, and optimizer state.
export NUM_GEN_NODES="${NUM_GEN_NODES:-4}"
export INFER_TP="${INFER_TP:-8}"
export INFER_EP="${INFER_EP:-8}"
export ENABLE_NSYS="${ENABLE_NSYS:-true}"
export MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL="${MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL:-100}"
# Needed to fit the model in memory with minimal model parallelism.
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-true}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
export OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-true}"
# No BF16 or FP8 optimizer state allowed for actual NT training.
export USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-false}"
NUM_TRAIN_NODES=$((NUM_NODES - NUM_GEN_NODES))
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_NAME="${WANDB_NAME:-nt-omni-super-67b-${TASK}-${NUM_NODES}n${GPUS_PER_NODE}g-tr${NUM_TRAIN_NODES}n-gen${NUM_GEN_NODES}n-tp${POLICY_TP}ep${POLICY_EP}cp${POLICY_CP}-itp${INFER_TP}iep${INFER_EP}-seq${MAX_SEQUENCE_LENGTH}-p${NUM_PROMPTS_PER_STEP}g${NUM_GENERATIONS_PER_PROMPT}-gbs${TRAIN_GBS}mbs${TRAIN_MICRO_BATCH_SIZE}-cpuoff${OPTIMIZER_CPU_OFFLOAD}}"
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

exec bash "${SCRIPT_DIR}/submit_nemotron_omni_multimodal_single_controller_8n4g.sh" "$@"
