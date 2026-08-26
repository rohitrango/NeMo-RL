#!/usr/bin/env bash
# Drive the Energon SFT pipeclean inside the container.
#   stage 1: loader-only iteration (fast, no Ray / no Megatron)
#   stage 2: full run_sft_v2
# Usage: STAGE=1|2|all bash tools/pipeclean.sh
set -uo pipefail

cd /opt/nemo-rl || cd "$(dirname "$0")/.."
CFG=examples/configs/sft_v2_tests/vlm_sft-nemotron-omni-30ba3b-4n8g-megatron-tp4etp4-super-test-blend.v1.yaml
STAGE=${STAGE:-all}
STEPS=${STEPS:-20}
OUT=${OUT:-/opt/nemo-rl/pipeclean-logs}
mkdir -p "$OUT"

if [[ "${SKIP_INIT:-0}" != "1" ]]; then
  echo "===== init deps ====="
  # Same set as code/RL/init.sh, but pinned to the container venv for every
  # package (the original omits --python on the last line).
  uv pip install --no-config --python /opt/nemo_rl_venv/bin/python \
      --index-url https://pypi.org/simple \
      "av>=17.1.0" "librosa==0.11.0" "soundfile>=0.13.1" soxr megatron-energon 2>&1 | tail -5
fi

echo "===== env ====="
echo "pwd=$(pwd)  host=$(hostname)"
python3 -c 'import sys; print("python:", sys.executable, sys.version.split()[0])' 2>&1
for m in ray megatron.energon av librosa soundfile; do
  python3 -c "import $m; print('  OK  $m', getattr($m,'__version__','?'))" 2>&1 | tail -1
done

if [[ "$STAGE" == "1" || "$STAGE" == "all" ]]; then
  echo "===== stage 1: loader-only ($STEPS steps) ====="
  uv run --extra energon --no-sync tools/iterate_energon_sft.py \
      --config "$CFG" --steps "$STEPS" \
      > "$OUT/loader.log" 2>&1
  echo "stage 1 exit=$?"
  tail -30 "$OUT/loader.log"
fi

if [[ "$STAGE" == "2" || "$STAGE" == "all" ]]; then
  echo "===== stage 2: full sft ====="
  uv run --extra energon --no-sync examples/run_sft_v2.py --config "$CFG" \
      > "$OUT/sft.log" 2>&1
  echo "stage 2 exit=$?"
  tail -60 "$OUT/sft.log"
fi
