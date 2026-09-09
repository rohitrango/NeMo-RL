#!/usr/bin/env bash
set -euo pipefail

# Eight-node/four-GPU NeMo-RL v2 SingleController launcher for Nemotron Omni.
# The default non-colocated layout matches the NeMo-RL v1 parity launchers:
# two training nodes and six Megatron generation nodes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTAINER_NEMORL="${CONTAINER_NEMORL:-/opt/nemo-rl}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${NEMORL}/workspace}"
TASK="${TASK:-clevr}"
COLOCATED="${COLOCATED:-false}"
ASYNC_GRPO="${ASYNC_GRPO:-true}"

if [[ "${COLOCATED}" != "false" ]]; then
  echo "SingleController currently requires COLOCATED=false." >&2
  exit 1
fi
if [[ "${ASYNC_GRPO}" != "true" ]]; then
  echo "SingleController uses async_rl and currently requires ASYNC_GRPO=true." >&2
  exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/cache/nemo-rl-omni}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${CACHE_ROOT}/megatron-checkpoints}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"

NUM_NODES="${NUM_NODES:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_GEN_NODES="${NUM_GEN_NODES:-6}"
GEN_GPUS_PER_NODE="${GEN_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
NUM_TRAIN_NODES=$((NUM_NODES - NUM_GEN_NODES))
TRAIN_WORLD_SIZE=$((NUM_TRAIN_NODES * GPUS_PER_NODE))
INFERENCE_WORLD_SIZE=$((NUM_GEN_NODES * GEN_GPUS_PER_NODE))
NUM_STORAGE_UNITS="${NUM_STORAGE_UNITS:-$((2 * NUM_NODES))}"

if (( NUM_NODES < 2 || NUM_GEN_NODES <= 0 || NUM_GEN_NODES >= NUM_NODES )); then
  echo "Non-colocated mode requires 0 < NUM_GEN_NODES < NUM_NODES." >&2
  exit 1
fi
if (( GEN_GPUS_PER_NODE != GPUS_PER_NODE )); then
  echo "Multi-node generation must reserve complete GPU nodes." >&2
  exit 1
fi

POLICY_TP="${POLICY_TP:-8}"
POLICY_EP="${POLICY_EP:-8}"
POLICY_CP="${POLICY_CP:-1}"
if (( POLICY_CP > 1 )); then
  MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY="${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY:-$((POLICY_TP * POLICY_CP * 2))}"
else
  MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY="${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY:-${POLICY_TP}}"
fi
INFER_TP="${INFER_TP:-8}"
INFER_EP="${INFER_EP:-8}"
if (( POLICY_CP <= 0 ||
      TRAIN_WORLD_SIZE % (POLICY_TP * POLICY_CP) != 0 ||
      TRAIN_WORLD_SIZE % POLICY_EP != 0 )); then
  echo "Training world size ${TRAIN_WORLD_SIZE} must be divisible by POLICY_TP*POLICY_CP and POLICY_EP." >&2
  exit 1
fi
if (( INFERENCE_WORLD_SIZE % INFER_TP != 0 || INFERENCE_WORLD_SIZE % INFER_EP != 0 )); then
  echo "Inference world size ${INFERENCE_WORLD_SIZE} must be divisible by INFER_TP and INFER_EP." >&2
  exit 1
fi
TRAIN_DP_SIZE=$((TRAIN_WORLD_SIZE / (POLICY_TP * POLICY_CP)))
INFERENCE_DP_SIZE=$((INFERENCE_WORLD_SIZE / INFER_TP))
TRAIN_MODEL_PARALLEL_SIZE=$((POLICY_TP * POLICY_CP))
if (( POLICY_EP > TRAIN_MODEL_PARALLEL_SIZE )); then
  TRAIN_MODEL_PARALLEL_SIZE=${POLICY_EP}
fi
INFERENCE_MODEL_PARALLEL_SIZE=${INFER_TP}
if (( INFER_EP > INFERENCE_MODEL_PARALLEL_SIZE )); then
  INFERENCE_MODEL_PARALLEL_SIZE=${INFER_EP}
fi
MAX_MODEL_PARALLEL_SIZE=${TRAIN_MODEL_PARALLEL_SIZE}
if (( INFERENCE_MODEL_PARALLEL_SIZE > MAX_MODEL_PARALLEL_SIZE )); then
  MAX_MODEL_PARALLEL_SIZE=${INFERENCE_MODEL_PARALLEL_SIZE}
fi
SEGMENT_SIZE="${SEGMENT_SIZE:-$(((MAX_MODEL_PARALLEL_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE))}"
if (( SEGMENT_SIZE <= 0 ||
      NUM_NODES % SEGMENT_SIZE != 0 ||
      NUM_TRAIN_NODES % SEGMENT_SIZE != 0 )); then
  echo "SEGMENT_SIZE ${SEGMENT_SIZE} must divide NUM_NODES ${NUM_NODES} and training nodes ${NUM_TRAIN_NODES}." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
MAX_STEPS="${MAX_STEPS:-1000000}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-$((INFERENCE_DP_SIZE * 2))}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
TRAIN_GBS="${TRAIN_GBS:-$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
if (( TRAIN_MICRO_BATCH_SIZE <= 0 )); then
  echo "TRAIN_MICRO_BATCH_SIZE must be positive." >&2
  exit 1
fi
if (( TRAIN_GBS % TRAIN_DP_SIZE != 0 )); then
  echo "TRAIN_GBS ${TRAIN_GBS} must be divisible by training DP size ${TRAIN_DP_SIZE}." >&2
  exit 1
fi
MAX_LOOKAHEAD_VERSIONS="${MAX_LOOKAHEAD_VERSIONS:-1}"
MAX_INFLIGHT_PROMPTS="${MAX_INFLIGHT_PROMPTS:-$((NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1)))}"
MAX_BUFFERED_ROLLOUTS="${MAX_BUFFERED_ROLLOUTS:-$((NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1)))}"
ASYNC_RL_DIAGNOSTICS="${ASYNC_RL_DIAGNOSTICS:-false}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
MONITOR_GPUS="${MONITOR_GPUS:-${WANDB_ENABLED}}"
GPU_MONITORING_COLLECTION_INTERVAL="${GPU_MONITORING_COLLECTION_INTERVAL:-10}"
GPU_MONITORING_FLUSH_INTERVAL="${GPU_MONITORING_FLUSH_INTERVAL:-10}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
REFIT_BACKEND="${REFIT_BACKEND:-nccl}"
BUFFER_SIZE_GB="${BUFFER_SIZE_GB:-8}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-false}"
OVERLAP_CPU_OPTIMIZER_D2H_H2D="${OVERLAP_CPU_OPTIMIZER_D2H_H2D:-false}"
OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
if [[ "${OPTIMIZER_CPU_OFFLOAD}" == "true" ]]; then
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
else
  OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-0.0}"
fi
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-true}"
EXP_AVG_DTYPE="${EXP_AVG_DTYPE:-bfloat16}"
EXP_AVG_SQ_DTYPE="${EXP_AVG_SQ_DTYPE:-bfloat16}"
STORE_PARAM_REMAINDERS="${STORE_PARAM_REMAINDERS:-true}"
if [[ "${USE_PRECISION_AWARE_OPTIMIZER}" == "true" ]]; then
  OPTIMIZER_PRECISION_OVERRIDES="\
++policy.megatron_cfg.optimizer.use_precision_aware_optimizer=true \
++policy.megatron_cfg.optimizer.exp_avg_dtype=${EXP_AVG_DTYPE} \
++policy.megatron_cfg.optimizer.exp_avg_sq_dtype=${EXP_AVG_SQ_DTYPE} \
++policy.megatron_cfg.optimizer.store_param_remainders=${STORE_PARAM_REMAINDERS}"
else
  OPTIMIZER_PRECISION_OVERRIDES="\
++policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false \
++policy.megatron_cfg.optimizer.exp_avg_dtype=float32 \
++policy.megatron_cfg.optimizer.exp_avg_sq_dtype=float32 \
++policy.megatron_cfg.optimizer.store_param_remainders=false"
fi
MEGATRON_ENABLE_CHUNKED_PREFILL="${MEGATRON_ENABLE_CHUNKED_PREFILL:-true}"
MEGATRON_TRANSFORMER_IMPL="${MEGATRON_TRANSFORMER_IMPL:-inference_optimized}"
MEGATRON_CUDA_GRAPH_IMPL="${MEGATRON_CUDA_GRAPH_IMPL:-local}"
MEGATRON_CUDA_GRAPH_SCOPE="${MEGATRON_CUDA_GRAPH_SCOPE:-block}"
MEGATRON_NUM_CUDA_GRAPHS="${MEGATRON_NUM_CUDA_GRAPHS:--1}"
MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE="${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE:-false}"
MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL="${MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL:-100}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
PREFIX_CACHING_MAMBA_GB="${PREFIX_CACHING_MAMBA_GB:-20}"
PREFIX_CACHING_EVICTION_POLICY="${PREFIX_CACHING_EVICTION_POLICY:-lru}"
PREFIX_CACHING_COORDINATOR_POLICY="${PREFIX_CACHING_COORDINATOR_POLICY:-longest_prefix}"
MAMBA_INFERENCE_SSM_STATES_DTYPE="${MAMBA_INFERENCE_SSM_STATES_DTYPE:-float32}"
OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-true}"
if [[ "${MEGATRON_TRANSFORMER_IMPL}" != "inference_optimized" &&
      "${MEGATRON_CUDA_GRAPH_IMPL}" == "local" && "${INFER_EP}" -gt 1 ]]; then
  MOE_PAD_EXPERTS_FOR_CG="${MOE_PAD_EXPERTS_FOR_CG:-true}"
else
  MOE_PAD_EXPERTS_FOR_CG=false
fi

TASK_VIDEO_EXPORTS=""
case "${TASK}" in
  clevr)
    CONFIG="${CONFIG:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-8n4g-megatron-single-controller-async.v1.yaml}"
    MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-4096}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
    # SingleController currently rejects validation during setup.
    VAL_PERIOD="${VAL_PERIOD:-0}"
    VAL_AT_START="${VAL_AT_START:-false}"
    VAL_AT_END="${VAL_AT_END:-false}"
    VAL_GBS="${VAL_GBS:-64}"
    VAL_SIZE="${VAL_SIZE:-64}"
    VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-536870912}"
    OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-true}"
    TASK_ENV="CLEVR_DATASET_MODE=config"
    WANDB_NAME_SUFFIX="nrl_v2_sctq_image"
    TASK_OVERRIDES=""
    ;;
  vstat)
    CONFIG="${CONFIG:-examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-video.v1.yaml}"
    MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-8192}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
    # SingleController currently rejects validation during setup.
    VAL_PERIOD="${VAL_PERIOD:-0}"
    VAL_AT_START="${VAL_AT_START:-false}"
    VAL_AT_END="${VAL_AT_END:-false}"
    VAL_GBS="${VAL_GBS:-2}"
    VAL_SIZE="${VAL_SIZE:-2}"
    DATA_ROOT="${DATA_ROOT:-${CONTAINER_NEMORL}/workspace/datasets/vstat-8n4g}"
    NEMO_RL_VIDEO_TRAIN_JSONL="${NEMO_RL_VIDEO_TRAIN_JSONL:-${DATA_ROOT}/train-gym.jsonl}"
    NEMO_RL_VIDEO_VAL_JSONL="${NEMO_RL_VIDEO_VAL_JSONL:-${DATA_ROOT}/val-gym.jsonl}"
    NEMO_RL_VIDEO_MEDIA_ROOT="${NEMO_RL_VIDEO_MEDIA_ROOT:-${DATA_ROOT}/media}"
    HF_DATASET="${HF_DATASET:-ShushengYang/VSTAT}"
    NUM_DATA_ROWS="${NUM_DATA_ROWS:-256}"
    NUM_FRAMES="${NUM_FRAMES:-16}"
    TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-2}"
    VIDEO_TARGET_PATCHES="${VIDEO_TARGET_PATCHES:-1024}"
    MIN_GENERATION_TOKENS="${MIN_GENERATION_TOKENS:-2000}"
    VISION_EMBEDDING_CACHE_MAX_BYTES="${VISION_EMBEDDING_CACHE_MAX_BYTES:-536870912}"
    OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-true}"
    ENABLE_THINKING="${ENABLE_THINKING:-true}"
    PREPARE_VSTAT="${PREPARE_VSTAT:-false}"
    WANDB_NAME_SUFFIX="nrl_v2_sctq_video"
    TASK_VIDEO_EXPORTS="\
export NRL_VIDEO_BACKEND=${NRL_VIDEO_BACKEND:-torchcodec}
export NRL_VIDEO_SAMPLING_STYLE=${NRL_VIDEO_SAMPLING_STYLE:-nemotron_vl}
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=${TEMPORAL_PATCH_SIZE}
export VLLM_VIDEO_LOADER_BACKEND=${VLLM_VIDEO_LOADER_BACKEND:-nemotron_vl}
export NEMO_RL_VIDEO_TRAIN_JSONL=${NEMO_RL_VIDEO_TRAIN_JSONL}
export NEMO_RL_VIDEO_VAL_JSONL=${NEMO_RL_VIDEO_VAL_JSONL}
export NEMO_RL_VIDEO_MEDIA_ROOT=${NEMO_RL_VIDEO_MEDIA_ROOT}"
    TASK_ENV="DATA_ROOT=${DATA_ROOT} HF_DATASET=${HF_DATASET} NUM_DATA_ROWS=${NUM_DATA_ROWS} NUM_FRAMES=${NUM_FRAMES} TEMPORAL_PATCH_SIZE=${TEMPORAL_PATCH_SIZE} VIDEO_TARGET_PATCHES=${VIDEO_TARGET_PATCHES} MIN_GENERATION_TOKENS=${MIN_GENERATION_TOKENS} ENABLE_THINKING=${ENABLE_THINKING} PREPARE_VSTAT=${PREPARE_VSTAT}"
    TASK_OVERRIDES="\
policy.tokenizer.chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
policy.megatron_cfg.env_vars.TORCH_CUDA_ARCH_LIST=\"'${TORCH_CUDA_ARCH_LIST:-10.0}'\" \
policy.megatron_cfg.freeze_vision_model=false \
policy.megatron_cfg.freeze_vision_projection=false \
policy.megatron_cfg.freeze_moe_router=false \
policy.megatron_cfg.mtp_num_layers=0 \
policy.megatron_cfg.mtp_use_repeated_layer=true \
policy.megatron_cfg.mtp_detach_heads=true \
policy.megatron_cfg.mtp_loss_scaling_factor=0.0 \
policy.megatron_cfg.pipeline_model_parallel_size=1 \
policy.megatron_cfg.moe_shared_expert_overlap=false \
policy.megatron_cfg.radio_force_cpe_eval_mode=true \
policy.megatron_cfg.clear_memory_caches_before_refit=true \
policy.megatron_cfg.optimizer.params_dtype=float32 \
policy.generation.mcore_generation_config.parsers=[nemotron-v3-reasoning,qwen3-coder-tool] \
++policy.generation.mcore_generation_config.video_num_frames=${NUM_FRAMES} \
++policy.generation.mcore_generation_config.video_temporal_patch_size=${TEMPORAL_PATCH_SIZE} \
++policy.generation.mcore_generation_config.video_target_num_patches=${VIDEO_TARGET_PATCHES} \
data.max_input_seq_length=${MAX_SEQUENCE_LENGTH} \
++data.default.num_frames=${NUM_FRAMES} \
++data.default.video_sampling_style=nemotron_vl \
++data.default.video_temporal_patch_size=${TEMPORAL_PATCH_SIZE} \
++data.default.min_generation_tokens=${MIN_GENERATION_TOKENS} \
++data.default.video_target_num_patches=${VIDEO_TARGET_PATCHES} \
data.default.video_maintain_aspect_ratio=true \
data.train.data_path=${NEMO_RL_VIDEO_TRAIN_JSONL} \
data.validation.data_path=${NEMO_RL_VIDEO_VAL_JSONL} \
++env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
grpo.deduplicate_multimodal_data=false"
    ;;
  *)
    echo "TASK must be clevr or vstat (got ${TASK})." >&2
    exit 1
    ;;
esac

INFERENCE_MAX_TOKENS="${INFERENCE_MAX_TOKENS:-$((MAX_SEQUENCE_LENGTH < 4096 ? MAX_SEQUENCE_LENGTH : 4096))}"
if (( TRAIN_GBS != NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT )); then
  echo "TRAIN_GBS must equal NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT." >&2
  exit 1
fi
if (( INFERENCE_MAX_TOKENS <= 0 )); then
  echo "INFERENCE_MAX_TOKENS must be positive." >&2
  exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/nemo-rl-v2-omni/${TASK}-8n4g}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"
JOB_NAME="${JOB_NAME:-nemotron-omni-${TASK}-single-controller-8n4g}"
EXP_NAME="${EXP_NAME:-${JOB_NAME}}"
PRECISION_RECIPE="${PRECISION_RECIPE:-bf16}"
WANDB_PROJ="${WANDB_PROJ:-mllm-rl-dev}"
WANDB_GROUP="${WANDB_GROUP:-adlr}"
WANDB_NAME="${WANDB_NAME:-${EXP_NAME}-${PRECISION_RECIPE}-${WANDB_NAME_SUFFIX}}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/asolergibert/RL/images/nemo-rl-nightly-gym.sqsh}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_mcore}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch_long}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
SBATCH_RESERVATION="${SBATCH_RESERVATION:-}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${RESULTS_DIR}/slurm}"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${NRL_MEGATRON_CHECKPOINT_DIR}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${SLURM_LOG_DIR}"
if [[ ! -f "${CONTAINER}" ]]; then
  echo "Container image does not exist: ${CONTAINER}" >&2
  exit 1
fi
if [[ ! -f "${NEMORL}/ray.sub" || ! -f "${NEMORL}/${CONFIG}" ||
      ! -f "${NEMORL}/examples/run_grpo_single_controller.py" ]]; then
  echo "NeMo-RL launcher, config, or entrypoint is missing under ${NEMORL}." >&2
  exit 1
fi
if [[ "${TASK}" == "vstat" && ! -f "${NEMORL}/scripts/prepare_nemotron_omni_vstat.py" ]]; then
  echo "VSTAT preparation script is missing under ${NEMORL}." >&2
  exit 1
fi

export NUM_NODES GPUS_PER_NODE CONTAINER
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}"
export NEMO_GYM_EXTRA_ROOTS="${NEMO_GYM_EXTRA_ROOTS:-${CONTAINER_NEMORL}/3rdparty/Gym-workspace/Gym}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "${TASK}" == "vstat" ]]; then
  export NRL_VIDEO_BACKEND="${NRL_VIDEO_BACKEND:-torchcodec}"
  export NRL_VIDEO_SAMPLING_STYLE="${NRL_VIDEO_SAMPLING_STYLE:-nemotron_vl}"
  export NRL_VIDEO_TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE}"
  export VLLM_VIDEO_LOADER_BACKEND="${VLLM_VIDEO_LOADER_BACKEND:-nemotron_vl}"
  export NEMO_RL_VIDEO_TRAIN_JSONL
  export NEMO_RL_VIDEO_VAL_JSONL
  export NEMO_RL_VIDEO_MEDIA_ROOT
fi

ENABLE_NSYS="${ENABLE_NSYS:-false}"
if [[ "${ENABLE_NSYS}" == "true" ]]; then
  export NRL_NSYS_WORKER_PATTERNS="${NRL_NSYS_WORKER_PATTERNS:-*policy*,*megatron*}"
  export NRL_NSYS_PROFILE_STEP_RANGE="${NRL_NSYS_PROFILE_STEP_RANGE:-2:5}"
  export LD_LIBRARY_PATH="/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/aarch64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export NRL_NSYS_EXTRA_OPTIONS="${NRL_NSYS_EXTRA_OPTIONS:-}"
else
  unset NRL_NSYS_WORKER_PATTERNS
  unset NRL_NSYS_PROFILE_STEP_RANGE
  unset NRL_NSYS_EXTRA_OPTIONS
fi

export SETUP_COMMAND=""
if [[ "${TASK}" == "vstat" ]]; then
  # NOTE: Remove the rm when the RL nightly container is unscrewed
  # and correctly installs the new Gym venv's.
  export SETUP_COMMAND="rm -rf \
  /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym \
  /opt/gym_venvs
cd ${CONTAINER_NEMORL} && bash tools/install_audio_deps.sh"
fi

DRIVER_TASK_SETUP=""
if [[ "${TASK}" == "vstat" ]]; then
  MEGATRON_WORKER_PYTHON="${RAY_MEGATRON_PYTHON:-${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
  DRIVER_TASK_SETUP="\
mkdir -p ${DATA_ROOT}
if [[ ! -x ${MEGATRON_WORKER_PYTHON} ]]; then
  echo 'Creating the Megatron worker environment before installing PyAV'
  FORCE_REBUILD_VENV=${NRL_FORCE_REBUILD_VENVS} uv run --no-sync python -c 'import os; from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES; from nemo_rl.utils.venvs import create_local_venv; create_local_venv(PY_EXECUTABLES.MCORE, \"nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker\", force_rebuild=os.environ[\"FORCE_REBUILD_VENV\"].lower() == \"true\")'
  export NRL_FORCE_REBUILD_VENVS=false
fi
if ! python -c 'import torchcodec' >/dev/null 2>&1 || ! ${MEGATRON_WORKER_PYTHON} -c 'import av' >/dev/null 2>&1; then
  RAY_MEGATRON_PYTHON=${MEGATRON_WORKER_PYTHON} bash tools/install_audio_deps.sh
fi
if [[ ${PREPARE_VSTAT} == true || ! -s ${NEMO_RL_VIDEO_TRAIN_JSONL} || ! -s ${NEMO_RL_VIDEO_VAL_JSONL} ]]; then
  uv run --no-sync python scripts/prepare_nemotron_omni_vstat.py --output-dir ${DATA_ROOT} --repo-id ${HF_DATASET} --num-rows ${NUM_DATA_ROWS}
fi"
fi

# Build the multi-node driver directly. Keep all runtime setup and Hydra
# overrides here so this launcher does not depend on the one-node smoke script.
export COMMAND="\
set -euo pipefail
NRL_SLURM_JOB_ID=\$(basename \"\$(dirname \"\$0\")\")
NRL_SLURM_JOB_ID=\${NRL_SLURM_JOB_ID%%-*}
cd ${CONTAINER_NEMORL}
export TASK=${TASK}
export NEMORL=${CONTAINER_NEMORL}
export WORKSPACE_ROOT=${CONTAINER_NEMORL}/workspace
if [[ -n \"\${NRL_NSYS_WORKER_PATTERNS:-}\" ]]; then
  NSYS_OUTPUT_DIR=\${WORKSPACE_ROOT}/nsys/\${NRL_SLURM_JOB_ID}
  mkdir -p \"\${NSYS_OUTPUT_DIR}\"
  if [[ -z \"\${NRL_NSYS_EXTRA_OPTIONS:-}\" ]]; then
    printf -v NRL_NSYS_EXTRA_OPTIONS '{\"o\":\"%s/%%p\",\"cpuctxsw\":\"none\",\"force-overwrite\":\"true\"}' \"\${NSYS_OUTPUT_DIR}\"
    export NRL_NSYS_EXTRA_OPTIONS
  fi
fi
export CONFIG=${CONFIG}
export MODEL_NAME=${MODEL_NAME}
export HF_HOME=${HF_HOME}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}
export HF_HUB_CACHE=${HF_HUB_CACHE}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
export HF_MODULES_CACHE=${HF_MODULES_CACHE}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
export NRL_MEGATRON_CHECKPOINT_DIR=${NRL_MEGATRON_CHECKPOINT_DIR}
export XDG_CACHE_HOME=${XDG_CACHE_HOME}
export TORCH_HOME=${TORCH_HOME}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR}
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR}
export NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR}
export NEMO_GYM_EXTRA_ROOTS=${NEMO_GYM_EXTRA_ROOTS}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN}
export NCCL_DEBUG=${NCCL_DEBUG}
${TASK_VIDEO_EXPORTS}
BRIDGE=${CONTAINER_NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
export PYTHONPATH=${CONTAINER_NEMORL}:\${NEMO_GYM_EXTRA_ROOTS}:\${BRIDGE}/src:\${BRIDGE}/3rdparty/Megatron-LM\${PYTHONPATH:+:\${PYTHONPATH}}
mkdir -p ${HF_HOME} ${NRL_MEGATRON_CHECKPOINT_DIR} ${RESULTS_DIR}
${DRIVER_TASK_SETUP}
exec env ${TASK_ENV} uv run --no-sync python examples/run_grpo_single_controller.py \
--config ${CONFIG} \
policy.model_name=${MODEL_NAME} \
policy.tokenizer.name=${MODEL_NAME} \
policy.is_vlm=true \
policy.max_total_sequence_length=${MAX_SEQUENCE_LENGTH} \
policy.train_global_batch_size=${TRAIN_GBS} \
policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE} \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
cluster.segment_size=${SEGMENT_SIZE} \
policy.megatron_cfg.tensor_model_parallel_size=${POLICY_TP} \
policy.megatron_cfg.expert_model_parallel_size=${POLICY_EP} \
policy.megatron_cfg.expert_tensor_parallel_size=1 \
policy.megatron_cfg.context_parallel_size=${POLICY_CP} \
policy.make_sequence_length_divisible_by=${MAKE_SEQUENCE_LENGTH_DIVISIBLE_BY} \
policy.megatron_cfg.sequence_parallel=true \
policy.megatron_cfg.bias_activation_fusion=false \
policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=${OVERLAP_GRAD_REDUCE} \
policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=${OVERLAP_PARAM_GATHER} \
policy.megatron_cfg.optimizer.optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD} \
policy.megatron_cfg.optimizer.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
policy.megatron_cfg.optimizer.overlap_cpu_optimizer_d2h_h2d=${OVERLAP_CPU_OPTIMIZER_D2H_H2D} \
${OPTIMIZER_PRECISION_OVERRIDES} \
policy.offload_optimizer_for_logprob=${OFFLOAD_OPTIMIZER_FOR_LOGPROB} \
policy.generation.backend=megatron \
++policy.generation.stop_strings=null \
++policy.generation.bad_words=null \
policy.generation.colocated.enabled=false \
policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
policy.generation.colocated.resources.gpus_per_node=${GEN_GPUS_PER_NODE} \
policy.generation.max_new_tokens=${MAX_NEW_TOKENS} \
policy.generation.mcore_generation_config.tensor_model_parallel_size=${INFER_TP} \
policy.generation.mcore_generation_config.expert_model_parallel_size=${INFER_EP} \
++policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
++policy.generation.mcore_generation_config.context_parallel_size=1 \
++policy.generation.mcore_generation_config.moe_router_dtype=fp32 \
policy.generation.mcore_generation_config.transformer_impl=${MEGATRON_TRANSFORMER_IMPL} \
policy.generation.mcore_generation_config.sequence_parallel=true \
policy.generation.mcore_generation_config.enable_chunked_prefill=${MEGATRON_ENABLE_CHUNKED_PREFILL} \
policy.generation.mcore_generation_config.enable_prefix_caching=${ENABLE_PREFIX_CACHING} \
++policy.generation.mcore_generation_config.prefix_caching_mamba_gb=${PREFIX_CACHING_MAMBA_GB} \
++policy.generation.mcore_generation_config.prefix_caching_eviction_policy=${PREFIX_CACHING_EVICTION_POLICY} \
++policy.generation.mcore_generation_config.prefix_caching_coordinator_policy=${PREFIX_CACHING_COORDINATOR_POLICY} \
++policy.generation.mcore_generation_config.vision_embedding_cache_max_bytes=${VISION_EMBEDDING_CACHE_MAX_BYTES} \
policy.generation.mcore_generation_config.cuda_graph_impl=${MEGATRON_CUDA_GRAPH_IMPL} \
policy.generation.mcore_generation_config.inference_cuda_graph_scope=${MEGATRON_CUDA_GRAPH_SCOPE} \
policy.generation.mcore_generation_config.num_cuda_graphs=${MEGATRON_NUM_CUDA_GRAPHS} \
policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE} \
++policy.generation.mcore_generation_config.logging_step_interval=${MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL} \
policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference=${MOE_PAD_EXPERTS_FOR_CG} \
policy.generation.mcore_generation_config.refit_backend=${REFIT_BACKEND} \
policy.generation.mcore_generation_config.buffer_size_gb=${BUFFER_SIZE_GB} \
++policy.generation.mcore_generation_config.kv_cache_management_mode=persist \
++policy.generation.mcore_generation_config.async_sched_mode=async \
++policy.generation.mcore_generation_config.logprobs_mode=raw_logprobs \
++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=${MAMBA_INFERENCE_SSM_STATES_DTYPE} \
++policy.generation.mcore_generation_config.mamba_inference_conv_states_dtype=float32 \
policy.generation.mcore_generation_config.max_model_len=${MAX_SEQUENCE_LENGTH} \
policy.generation.mcore_generation_config.max_tokens=${INFERENCE_MAX_TOKENS} \
++data_plane.enabled=true \
++data_plane.impl=transfer_queue \
++data_plane.backend=simple \
++data_plane.claim_meta_poll_interval_s=0.5 \
++data_plane.simple.num_storage_units=${NUM_STORAGE_UNITS} \
grpo.async_grpo=null \
grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
grpo.max_num_steps=${MAX_STEPS} \
grpo.val_period=${VAL_PERIOD} \
grpo.val_at_start=${VAL_AT_START} \
grpo.val_at_end=${VAL_AT_END} \
grpo.val_batch_size=${VAL_GBS} \
grpo.max_val_samples=${VAL_SIZE} \
grpo.overlong_filtering=false \
loss_fn.use_importance_sampling_correction=true \
++async_rl.sampler.name=in_order \
++async_rl.sampler.max_lookahead_versions=${MAX_LOOKAHEAD_VERSIONS} \
++async_rl.recompute_kv_cache_after_weight_updates=false \
++async_rl.min_groups_for_streaming_train=${NUM_PROMPTS_PER_STEP} \
++async_rl.max_inflight_prompts=${MAX_INFLIGHT_PROMPTS} \
++async_rl.max_buffered_rollouts=${MAX_BUFFERED_ROLLOUTS} \
++async_rl.diagnostics=${ASYNC_RL_DIAGNOSTICS} \
checkpointing.enabled=${CHECKPOINTING_ENABLED} \
checkpointing.checkpoint_dir=${RESULTS_DIR} \
checkpointing.metric_name=null \
logger.log_dir=${RESULTS_DIR} \
logger.wandb_enabled=${WANDB_ENABLED} \
logger.tensorboard_enabled=false \
logger.monitor_gpus=${MONITOR_GPUS} \
logger.gpu_monitoring.collection_interval=${GPU_MONITORING_COLLECTION_INTERVAL} \
logger.gpu_monitoring.flush_interval=${GPU_MONITORING_FLUSH_INTERVAL} \
logger.wandb.name=${WANDB_NAME}-\${NRL_SLURM_JOB_ID} \
logger.wandb.project=${WANDB_PROJ} \
+logger.wandb.entity=${WANDB_GROUP} \
${TASK_OVERRIDES} \
${EXTRA_OVERRIDES}"

echo "Submitting ${JOB_NAME}: ${NUM_NODES}x${GPUS_PER_NODE}"
echo "  split: ${NUM_TRAIN_NODES} train nodes / ${NUM_GEN_NODES} generation nodes"
echo "  train TP/EP/CP/world/DP: ${POLICY_TP}/${POLICY_EP}/${POLICY_CP}/${TRAIN_WORLD_SIZE}/${TRAIN_DP_SIZE}"
echo "  generation TP/EP/world/DP: ${INFER_TP}/${INFER_EP}/${INFERENCE_WORLD_SIZE}/${INFERENCE_DP_SIZE}"
echo "  max model parallel GPUs / segment nodes: ${MAX_MODEL_PARALLEL_SIZE}/${SEGMENT_SIZE}"
echo "  prompts/generations/train_gbs: ${NUM_PROMPTS_PER_STEP}/${NUM_GENERATIONS_PER_PROMPT}/${TRAIN_GBS}"
echo "  async sampler/lookahead/inflight/buffer: in_order/${MAX_LOOKAHEAD_VERSIONS}/${MAX_INFLIGHT_PROMPTS}/${MAX_BUFFERED_ROLLOUTS}"
echo "  sequence/inference-step/new tokens: ${MAX_SEQUENCE_LENGTH}/${INFERENCE_MAX_TOKENS}/${MAX_NEW_TOKENS}"
echo "  generation: colocated=${COLOCATED} async=${ASYNC_GRPO} refit=${REFIT_BACKEND}"
echo "  Megatron: transformer=${MEGATRON_TRANSFORMER_IMPL} chunked_prefill=${MEGATRON_ENABLE_CHUNKED_PREFILL} prefix_caching=${ENABLE_PREFIX_CACHING} logging_interval=${MEGATRON_INFERENCE_LOGGING_STEP_INTERVAL}"
echo "  CUDA graphs: impl=${MEGATRON_CUDA_GRAPH_IMPL} scope=${MEGATRON_CUDA_GRAPH_SCOPE} count=${MEGATRON_NUM_CUDA_GRAPHS} non_decode=${MEGATRON_USE_CUDA_GRAPHS_FOR_NON_DECODE} moe_padding=${MOE_PAD_EXPERTS_FOR_CG}"
echo "  optimizer: cpu_offload=${OPTIMIZER_CPU_OFFLOAD} offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} overlap_d2h_h2d=${OVERLAP_CPU_OPTIMIZER_D2H_H2D} logprob_offload=${OFFLOAD_OPTIMIZER_FOR_LOGPROB}"
echo "  config: ${CONFIG}"
if [[ "${TASK}" == "vstat" ]]; then
  echo "  VSTAT: repo=${HF_DATASET} rows=${NUM_DATA_ROWS} prepare=${PREPARE_VSTAT}"
  echo "  datasets: train=${NEMO_RL_VIDEO_TRAIN_JSONL} val=${NEMO_RL_VIDEO_VAL_JSONL} media=${NEMO_RL_VIDEO_MEDIA_ROOT}"
fi
echo "  NSYS: enabled=${ENABLE_NSYS}${NRL_NSYS_PROFILE_STEP_RANGE:+ step_range=${NRL_NSYS_PROFILE_STEP_RANGE}}"
echo "  W&B: ${WANDB_GROUP}/${WANDB_PROJ}/${WANDB_NAME}-<slurm-job-id> (enabled=${WANDB_ENABLED})"
echo "  GPU monitoring: enabled=${MONITOR_GPUS} collection=${GPU_MONITORING_COLLECTION_INTERVAL}s flush=${GPU_MONITORING_FLUSH_INTERVAL}s"

SBATCH_ARGS=(
  --nodes="${NUM_NODES}"
  --account="${SBATCH_ACCOUNT}"
  --partition="${SBATCH_PARTITION}"
  --job-name="${JOB_NAME}"
  --time="${SBATCH_TIME}"
  --output="${SLURM_LOG_DIR}/%j.out"
  --error="${SLURM_LOG_DIR}/%j.out"
  --gres="gpu:${GPUS_PER_NODE}"
  --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"data_loading","description":"Async GRPO RL training: training GPUs idle during rollout collection (~30min) and validation each step"}}'
  --exclusive
  --mem=0
  --dependency=singleton
  --segment="${SEGMENT_SIZE}"
)
if [[ -n "${SBATCH_QOS}" ]]; then
  SBATCH_ARGS+=(--qos="${SBATCH_QOS}")
fi
if [[ -n "${SBATCH_RESERVATION}" ]]; then
  SBATCH_ARGS+=(--reservation="${SBATCH_RESERVATION}")
fi

BASE_LOG_DIR="${SLURM_LOG_DIR}" \
MOUNTS="${MOUNTS:-/lustre:/lustre},${NEMORL}:${CONTAINER_NEMORL}" \
sbatch "${SBATCH_ARGS[@]}" "${NEMORL}/ray.sub"
