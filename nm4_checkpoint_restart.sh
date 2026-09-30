#!/usr/bin/env bash
# Reproduce CLEVR save/restart runs (NM4 or Qwen2.5-VL-7B).
# Source jobs: Adam 4079404/4079411; Muon 4083225/4083242.
# The original jobs passed validation overrides; this version uses the current
# validation-free YAML configs.

set -euo pipefail

script_args=()
overrides=()
while (( $# )); do
    if [[ "$1" == -- ]]; then
        shift
        overrides=("$@")
        break
    fi
    script_args+=("$1")
    shift
done
set -- "${script_args[@]}"

if (( $# < 2 || $# > 4 )); then
    echo "Usage: $0 {adam|muon} {bf16|mxfp8} [nm4|qwen|qwen2_5_vl_7b_clevr_1n4g_tp2cp1.yaml] [run_id] [-- OVERRIDE ...]" >&2
    exit 2
fi

optimizer=$1
precision=$2
case "${3:-nm4}" in
    nm4)
        recipe=nm4
        run_id=${4:-$(date -u +%Y%m%dT%H%M%SZ)}
        ;;
    qwen|qwen2_5_vl_7b_clevr_1n4g_tp2cp1|qwen2_5_vl_7b_clevr_1n4g_tp2cp1.yaml|examples/configs/sft_v2_tests/qwen2_5_vl_7b_clevr_1n4g_tp2cp1.yaml)
        recipe=qwen
        run_id=${4:-$(date -u +%Y%m%dT%H%M%SZ)}
        ;;
    *)
        if (( $# == 4 )); then
            echo "Unknown recipe: $3" >&2
            exit 2
        fi
        recipe=nm4
        run_id=$3  # Preserve the original three-argument form.
        ;;
esac

case "$optimizer" in
    adam|muon) ;;
    *) echo "Unknown optimizer: $optimizer" >&2; exit 2 ;;
esac
case "$recipe:$precision" in
    nm4:bf16)
        config=examples/configs/sft_v2_tests/nm4_clevr_4n.yaml
        name_prefix=nt4-clevr
        ;;
    nm4:mxfp8)
        config=examples/configs/sft_v2_tests/nm4_clevr_4n_mxfp8.yaml
        name_prefix=nt4-clevr
        ;;
    qwen:bf16)
        config=examples/configs/sft_v2_tests/qwen2_5_vl_7b_clevr_1n4g_tp2cp1.yaml
        name_prefix=qwen2.5-vl-7b-clevr
        ;;
    qwen:mxfp8)
        echo "No Qwen MXFP8 recipe exists; use bf16 for this recipe." >&2
        exit 2
        ;;
    *) echo "Unknown precision: $precision" >&2; exit 2 ;;
esac
if [[ ! "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "run_id must contain only letters, digits, _ or -" >&2
    exit 2
fi

if [[ "$recipe" == nm4 ]]; then
    checkpoint_dir="/data/nemorl/rohit-sft_v2_up_n4/restart-${precision}-${optimizer}-${run_id}"
else
    checkpoint_dir="/data/nemorl/rohit-sft_v2_up_n4/restart-${recipe}-${precision}-${optimizer}-${run_id}"
fi
wandb_name="${name_prefix}-restart-${precision}-${optimizer}-${run_id}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/nemo-rl/venvs-fa4}"

# A fresh directory makes the first segment a pretrained start. Later segments
# must find the immediately preceding checkpoint and restore its optimizer.
if [[ -e "$checkpoint_dir" ]]; then
    echo "Checkpoint directory already exists: $checkpoint_dir" >&2
    exit 1
fi

echo "Running $recipe/$precision/$optimizer with checkpoint_dir=$checkpoint_dir"
previous_target=
for target in 20 50 75 100; do
    if [[ -n "$previous_target" ]]; then
        test -f "$checkpoint_dir/step_${previous_target}/training_info.json"
        export NEMO_RL_REQUIRE_RESUME=1
        export NEMO_RL_EXPECTED_RESUME_CHECKPOINT="$checkpoint_dir/step_${previous_target}"
    else
        unset NEMO_RL_REQUIRE_RESUME NEMO_RL_EXPECTED_RESUME_CHECKPOINT
    fi

    echo "$precision $optimizer: train to step $target; checkpoints: $checkpoint_dir"
    uv run --no-sync examples/run_sft_v2.py \
        --config "$config" \
        "sft.max_num_steps=$target" \
        "policy.megatron_cfg.optimizer.optimizer=$optimizer" \
        +policy.megatron_cfg.scheduler.max_steps=100 \
        checkpointing.enabled=true \
        checkpointing.save_period=100 \
        checkpointing.keep_top_k=5 \
        checkpointing.save_optimizer=true \
        "checkpointing.checkpoint_dir=$checkpoint_dir" \
        logger.wandb.project=rohit-sft_v2_up_n4 \
        "logger.wandb.name=$wandb_name" \
        "${overrides[@]}"
    test -f "$checkpoint_dir/step_${target}/training_info.json"
    previous_target=$target
done
