#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Build the Python environment used by tools/compare_cookers.py and
# tools/compare_task_encoders.py.
#
# Deliberately NOT the project's uv environment: the comparison also imports the
# Megatron-LM reference cookers and task encoder, and needs a torch/transformers
# pair that works on CPU only. megatron-core is not required -- the harness
# stubs the four modules it needs straight from the reference source.
#
# Usage:
#   bash tools/setup_parity_env.sh [ENV_DIR]
#
# ENV_DIR defaults to $PARITY_ENV or /tmp/cookenv2. Put it on shared storage
# when the environment has to be visible from a compute node; /tmp is
# node-local.
set -euo pipefail

ENV_DIR="${1:-${PARITY_ENV:-/tmp/cookenv2}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -x "${ENV_DIR}/bin/python" ]]; then
  echo "env already present: ${ENV_DIR}"
  "${ENV_DIR}/bin/python" - <<'PY'
import torch, transformers, megatron.energon, numpy
print(f"  torch {torch.__version__}  transformers {transformers.__version__}  numpy {numpy.__version__}")
PY
  exit 0
fi

echo "creating venv at ${ENV_DIR} using ${PYTHON_BIN}"
"${PYTHON_BIN}" -m venv "${ENV_DIR}"

# CPU wheels only. torch 2.6 + torchvision 0.21 is the pair that satisfies both
# transformers and the reference's torchvision imports.
"${ENV_DIR}/bin/pip" install -q --no-input --upgrade pip
"${ENV_DIR}/bin/pip" install -q --no-input \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  'torch==2.6.0' 'torchvision==0.21.0'

# megatron-energon[av-decode] brings av, needed to decode video.
# albumentations and einops are imported by the reference image pipeline.
# hydra-core/omegaconf are needed to load the NeMo-RL recipe yaml.
"${ENV_DIR}/bin/pip" install -q --no-input \
  'megatron-energon[av-decode]' \
  transformers datasets pydantic pyyaml pillow \
  albumentations einops hydra-core omegaconf

"${ENV_DIR}/bin/python" - <<'PY'
import torch, torchvision, transformers, numpy, megatron.energon, albumentations, einops
from megatron.energon import CachePool, FileStore, cooker, stateless  # noqa: F401
from megatron.energon.av import AVDecoder  # noqa: F401
print("environment OK")
print(f"  torch {torch.__version__}  torchvision {torchvision.__version__}")
print(f"  transformers {transformers.__version__}  numpy {numpy.__version__}")
PY

echo "done: ${ENV_DIR}"
