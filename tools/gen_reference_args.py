#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the Megatron args file used by tools/compare_task_encoders.py.

Hand-editing that file produced two false findings (a stale use_thumbnail=True
and an invented video_min_num_frames=1), so it is generated instead. Three
layers, applied in order:

  1. core defaults      megatron/training/arguments.py        (AST, no import)
  2. multimodal defaults examples/multimodal/multimodal_args.py (parser)
  3. TransformerConfig   megatron/core/transformer/transformer_config.py (AST)
  4. launch-script values, below, each with its line number

Layer 4 is the only hand-written part and every entry cites the production
script it came from:

    examples/multimodal/v3p5_super_prod_run/
      sft_from_super35_dualds_4k_1of8_recover_iter366_radio_v4_h_full_generalist.sh

Usage:
    python tools/gen_reference_args.py --reference-root energon-megatron-lm \
        --out /tmp/ref_args.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

# Values the production launch script passes explicitly. Anything not listed
# here keeps the harvested default -- including use_thumbnail, which the script
# never passes.
LAUNCH_SCRIPT_VALUES: dict[str, Any] = {
    # model / vision, lines 374-386
    "patch_dim": 16,
    "img_h": 512,
    "img_w": 512,
    "vision_model_type": "radio",
    "class_token_len": 10,
    "disable_vision_class_token": True,
    "image_tag_type": "internvl",
    "language_model_type": "nemotron6-super",
    "eod_mask_loss": True,
    "train_full_dataset": True,
    # tiling, lines 237-239
    "pixel_shuffle": True,
    "conv_merging": False,
    "dynamic_resolution": True,
    "dynamic_resolution_min_patches": 1024,
    "dynamic_resolution_max_patches": 13312,
    # tokenizer, lines 71, 447-451
    "tokenizer_type": "MultimodalTokenizer",
    "tokenizer_prompt_format": "nemotron6-moe",
    "thinking_trace_format": "ultra",  # line 450
    # video, lines 288-320
    "video_max_num_frames": 64,
    "video_target_num_patches": 1024,
    "video_aug_scale_frames_up": 4,
    "video_aug_scale_resolution_up": None,
    "video_aug_scale_resolution_only": True,
    "video_maintain_aspect_ratio": True,
    "separate_video_embedder": True,
    "video_temporal_patch_size": 2,
    "video_prompt_version": 2,
    "video_decode_thread_count": 0,
}

# Deliberate deviations from the launch script, for a single-rank comparison
# against the NeMo-RL recipe rather than against the production cluster run.
HARNESS_OVERRIDES: dict[str, Any] = {
    # The NeMo-RL recipe under test uses 262144, not the script's 524288
    # (line 183). Both stacks must budget identically or tiling diverges.
    "decoder_seq_length": 262144,
    "dataloader_seq_length": 262144,
    "packing_seq_length": 262144,
    # Packing is compared separately; off here.
    "packing_buffer_size": None,
    # Single rank: CP=8 (line 138) would only add sequence padding.
    "context_parallel_size": 1,
    "sequence_parallel": False,
}


def harvest_add_argument_defaults(path: Path) -> dict[str, Any]:
    """Read argparse defaults out of a source file without importing it."""
    defaults: dict[str, Any] = {}
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        flags = [
            a.value
            for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        long_flags = [f for f in flags if f.startswith("--")]
        if not long_flags:
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        dest = None
        if "dest" in keywords:
            try:
                dest = ast.literal_eval(keywords["dest"])
            except ValueError:
                dest = None
        if dest is None:
            dest = long_flags[0][2:].replace("-", "_")
        value = None
        if "default" in keywords:
            try:
                value = ast.literal_eval(keywords["default"])
            except ValueError:
                value = None
        elif "action" in keywords:
            try:
                action = ast.literal_eval(keywords["action"])
            except ValueError:
                action = None
            value = {"store_true": False, "store_false": True}.get(action)
        defaults[dest] = value
    return defaults


def harvest_dataclass_defaults(path: Path) -> dict[str, Any]:
    """Read annotated dataclass field defaults without importing."""
    defaults: dict[str, Any] = {}
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and statement.value is not None
                and isinstance(statement.target, ast.Name)
            ):
                try:
                    defaults[statement.target.id] = ast.literal_eval(statement.value)
                except ValueError:
                    continue
    return defaults


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="energon-megatron-lm")
    parser.add_argument("--out", default="/tmp/ref_args.json")
    args = parser.parse_args()

    root = Path(args.reference_root).expanduser().resolve()
    merged: dict[str, Any] = {}

    core = harvest_add_argument_defaults(root / "megatron/training/arguments.py")
    merged.update(core)

    transformer = harvest_dataclass_defaults(
        root / "megatron/core/transformer/transformer_config.py"
    )
    merged.update(transformer)

    # multimodal_args imports llava_model, so install the stubs first.
    import compare_task_encoders as harness

    harness.install_megatron_core_stubs(root, verbose=False)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "examples" / "multimodal"))
    from examples.multimodal.multimodal_args import add_multimodal_extra_args

    multimodal = {
        action.dest: action.default
        for action in add_multimodal_extra_args(argparse.ArgumentParser())._actions
        if action.dest != "help"
    }
    merged.update(multimodal)

    merged.update(LAUNCH_SCRIPT_VALUES)
    merged.update(HARNESS_OVERRIDES)
    merged["prompt_path"] = str(root / "examples/multimodal/manual_prompts.json")

    Path(args.out).write_text(json.dumps(merged, indent=2, default=str, sort_keys=True))
    print(f"wrote {args.out}")
    print(f"  core defaults        {len(core)}")
    print(f"  transformer defaults {len(transformer)}")
    print(f"  multimodal defaults  {len(multimodal)}")
    print(f"  launch-script values {len(LAUNCH_SCRIPT_VALUES)}")
    print(f"  harness overrides    {len(HARNESS_OVERRIDES)}")
    print(f"  total keys           {len(merged)}")
    for key in ("use_thumbnail", "video_min_num_frames", "thinking_trace_format",
                "tokenizer_prompt_format", "video_max_num_frames"):
        print(f"    {key:<24} {merged.get(key)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
