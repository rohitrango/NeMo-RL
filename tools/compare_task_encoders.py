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

"""Compare the NeMo-RL and Megatron-LM stacks end to end: cook, then pre-encode.

``tools/compare_cookers.py`` stops at the cooker. This runs the same rows one
step further and diffs what the task encoders produce, which is where the token
ids, the loss mask, and the visual sizing are decided.

    row -> cook (both) -> preencode (both) -> normalize -> diff

Per row the script records one verdict:

    identical | differ | both_raised | only_reference_raised | only_nemo_rl_raised

and, for a differing row, which of these fields disagree:

    tokens, loss_mask, trainable_tokens, image_count, image_sizes,
    num_frames, total_len

A row that fails at the cooker is reported with stage ``cook``; a row that
cooks on both sides and fails later is reported with stage ``encode``. That
split matters: the reference drops several classes of row at the encoder that
NeMo-RL rejects at the cooker, and vice versa.

Requirements beyond ``tools/compare_cookers.py``:

  * ``megatron-core`` — the reference encoder imports
    ``megatron.core.models.multimodal.llava_model``.
  * ``transformers`` — for the NeMo-RL processor.
  * a model directory with tokenizer + processor config (``--model-path``).
  * a JSON file of Megatron args for the reference encoder
    (``--reference-args``). Write ``--emit-reference-args-template`` to get a
    starting skeleton; the field set is cluster- and recipe-specific.

Run the NeMo-RL side alone with ``--skip-reference`` when megatron-core or the
args file is not available. The cooker stage still runs for both sides in that
mode, so the row accounting stays meaningful.

Usage:

    uv run --locked --extra mcore --extra energon tools/compare_task_encoders.py \
        --subset ~/data/super-test-blend/cook_subset_v2/subset.yaml \
        --config examples/configs/sft_v2_tests/vlm_sft-nemotron-omni-30ba3b-4n8g-megatron-tp4etp4-super-test-blend.v1.yaml \
        --model-path ~/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
        --reference-args /tmp/reference_args.json \
        --limit 25
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import compare_cookers as cc  # noqa: E402  (path set above)

# image_processing.py:18 imports `data_loading.conversation_sample` absolutely
# while cookers/conversation.py uses a relative import. With both the repo root
# and examples/multimodal on sys.path the same file loads twice under two names,
# so ImageMedia has two identities and isinstance() in compute_params fails.
# Import everything under the `data_loading.*` name the reference itself uses.
REFERENCE_MODULES = (
    "data_loading.cookers.conversation",
    "data_loading.cookers.granary",
    "data_loading.cookers.audio_conversation",
    "data_loading.cookers.omcat_legacy_audio_conversation",
)

POSTENCODE_FIELDS = (
    "post_length",
    "post_pixel_shapes",
    "post_pixel_values",
    "post_num_tiles",
    "post_num_frames",
)

ENCODE_FIELDS = (
    "final_tokens",
    "final_loss_mask",
    "final_length",
    "tokens",
    "loss_mask",
    "trainable_tokens",
    "image_count",
    "image_sizes",
    "num_frames",
    "total_len",
)

# Minimal Megatron arg skeleton for MultiModalTaskEncoder.__init__. The real set
# is recipe-specific; start here and add whatever the constructor asks for.
REFERENCE_ARGS_TEMPLATE = {
    # Values taken from the production launch script
    # examples/multimodal/v3p5_super_prod_run/
    #   sft_from_super35_dualds_4k_1of8_recover_iter366_radio_v4_h_full_generalist.sh
    "prompt_path": "energon-megatron-lm/examples/multimodal/manual_prompts.json",
    "patch_dim": 16,
    "img_h": 512,
    "img_w": 512,
    "vision_model_type": "radio",
    "class_token_len": 10,
    "disable_vision_class_token": True,
    "image_tag_type": "internvl",
    "language_model_type": "nemotron6-super",
    "pixel_shuffle": True,
    "conv_merging": False,
    "dynamic_resolution": True,
    "dynamic_resolution_min_patches": 1024,
    "dynamic_resolution_max_patches": 13312,
    "decoder_seq_length": 524288,
    "dataloader_seq_length": 524288,
    "packing_seq_length": 524288,
    "packing_buffer_size": 5000,
    "packing_knapsack_algorithm": "balanced_greedy_knapsack",
    "packing_algorithm_parameters": "balanced_knapsack_delta=5",
    "eod_mask_loss": True,
    "train_full_dataset": True,
}

# --tokenizer-type MultimodalTokenizer, with these flags from the same script.
REFERENCE_TOKENIZER_PROMPT_FORMAT = "nemotron6-moe"
REFERENCE_TOKENIZER_IMAGE_TAG_TYPE = "internvl"
REFERENCE_TOKENIZER_SPECIAL_TOKENS = [
    "<image>", "<img>", "</img>", "<quad>", "</quad>",
    "<ref>", "</ref>", "<box>", "</box>",
]


# The reference task encoder imports six names from megatron-core. Every one is
# a constant or a self-contained function, but the modules holding them drag in
# the distributed stack (llava_model -> distributed -> pipeline_parallel ->
# paged_stash -> triton). Loading the functions out of their source files and
# registering stub modules removes that chain without reimplementing anything:
# the code below is executed verbatim from the reference checkout.
#
# megatron.core.models.multimodal.llava_model    IGNORE_INDEX, SOUND_TOKEN, IMAGE_TOKEN
# megatron.core.models.vision.clip_vit_model     get_num_image_embeddings
# megatron.core.models.multimodal.utils          patchify_image
# megatron.core.models.multimodal.context_parallel  get_padding
# megatron.training                              get_args, get_tokenizer
#
# image_processing.py:247 also imports get_hf_model_type, but only inside the
# `vision_model_type.startswith("hf://")` branch. This recipe uses "radio", so
# that branch is never taken and the import never runs.
_STUBBED_MODULES = (
    "megatron.core.models.multimodal.llava_model",
    "megatron.core.models.vision.clip_vit_model",
    "megatron.core.models.multimodal.utils",
    "megatron.core.models.multimodal.context_parallel",
)


def _exec_module_functions(path: Path, extra_globals: dict[str, Any]) -> dict[str, Any]:
    """Execute a source file's top-level functions and constants, skipping imports.

    Nodes that fail because they reference something an import would have
    provided are skipped and reported, so nothing disappears quietly.
    """
    import ast
    import types

    tree = ast.parse(path.read_text(), filename=str(path))
    # dataclass resolves annotations through sys.modules[cls.__module__], so the
    # namespace has to be a registered module, not a bare dict.
    module_name = f"_reference_stub_{path.stem}"
    module = types.ModuleType(module_name)
    module.__dict__.update(extra_globals)
    sys.modules[module_name] = module
    namespace: dict[str, Any] = module.__dict__
    skipped: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        try:
            exec(  # noqa: S102 - executing the reference source is the point
                compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
                namespace,
            )
        except Exception as error:  # noqa: BLE001
            skipped.append(f"{path.name}:{getattr(node, 'lineno', '?')} {error}")
    namespace["__skipped__"] = skipped
    return namespace


def install_megatron_core_stubs(reference_root: Path, *, verbose: bool = True) -> None:
    """Register stub megatron-core modules built from the reference source."""
    import math
    import types
    import typing

    import torch

    core = reference_root / "megatron" / "core" / "models"
    shared = {
        "torch": torch,
        "math": math,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "List": typing.List,
        "Tuple": typing.Tuple,
    }

    clip = _exec_module_functions(core / "vision" / "clip_vit_model.py", shared)
    multimodal_utils = _exec_module_functions(core / "multimodal" / "utils.py", shared)
    context_parallel = _exec_module_functions(
        core / "multimodal" / "context_parallel.py", shared
    )

    llava_source = (core / "multimodal" / "llava_model.py").read_text()
    constants: dict[str, Any] = {}
    for name in (
        "IGNORE_INDEX",
        "SOUND_TOKEN",
        "IMAGE_TOKEN",
        "DEFAULT_IMAGE_TOKEN_INDEX",
        "DEFAULT_SOUND_TOKEN_INDEX",
    ):
        import ast

        for node in ast.parse(llava_source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                try:
                    # -100 parses as UnaryOp(USub, Constant), not Constant.
                    constants[name] = ast.literal_eval(node.value)
                except ValueError:
                    continue
        if name not in constants:
            raise RuntimeError(f"could not read {name} from llava_model.py")

    exported = {
        "megatron.core.models.multimodal.llava_model": constants,
        "megatron.core.models.vision.clip_vit_model": {
            "get_num_image_embeddings": clip["get_num_image_embeddings"]
        },
        "megatron.core.models.multimodal.utils": {
            "patchify_image": multimodal_utils["patchify_image"]
        },
        "megatron.core.models.multimodal.context_parallel": {
            "get_padding": context_parallel["get_padding"]
        },
    }
    for dotted, members in exported.items():
        module = types.ModuleType(dotted)
        for key, value in members.items():
            setattr(module, key, value)
        sys.modules[dotted] = module

    if verbose:
        print("megatron-core stubs installed from the reference source:")
        for name, value in constants.items():
            print(f"  llava_model.{name} = {value!r}")
        for namespace in (clip, multimodal_utils, context_parallel):
            for entry in namespace["__skipped__"]:
                print(f"  skipped node: {entry}")


def load_reference_tokenizer_class(reference_root: Path) -> Any:
    """Load MegatronMultimodalTokenizer from source.

    Importing it normally pulls megatron.core.__init__ and the distributed
    stack, so the module is executed from its own file with the llava constants
    injected. The class body is the reference's, unmodified.
    """
    import dataclasses
    import json as _json
    import os
    import time
    import typing
    import uuid

    import numpy as np
    import transformers

    llava = sys.modules["megatron.core.models.multimodal.llava_model"]
    namespace = _exec_module_functions(
        reference_root
        / "megatron/core/tokenizers/vision/libraries/multimodal_tokenizer.py",
        {
            "json": _json,
            "os": os,
            "time": time,
            "uuid": uuid,
            "np": np,
            "transformers": transformers,
            "HAVE_TRANSFORMERS": True,
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "Any": typing.Any,
            "Dict": typing.Dict,
            "List": typing.List,
            "Optional": typing.Optional,
            "Union": typing.Union,
            "IGNORE_INDEX": llava.IGNORE_INDEX,
            "IMAGE_TOKEN": llava.IMAGE_TOKEN,
            "SOUND_TOKEN": llava.SOUND_TOKEN,
            "DEFAULT_IMAGE_TOKEN_INDEX": llava.DEFAULT_IMAGE_TOKEN_INDEX,
            "DEFAULT_SOUND_TOKEN_INDEX": llava.DEFAULT_SOUND_TOKEN_INDEX,
        },
    )
    tokenizer_class = namespace.get("MegatronMultimodalTokenizer")
    if tokenizer_class is None:
        raise RuntimeError(
            "MegatronMultimodalTokenizer did not load; skipped nodes: "
            + "; ".join(namespace["__skipped__"])
        )
    return tokenizer_class


def build_reference_tokenizer(reference_root: Path, tokenizer_path: Path) -> Any:
    """Instantiate the reference tokenizer with the production launch flags."""
    tokenizer_class = load_reference_tokenizer_class(reference_root)
    return tokenizer_class(
        path=str(tokenizer_path),
        prompt_format=REFERENCE_TOKENIZER_PROMPT_FORMAT,
        special_tokens=REFERENCE_TOKENIZER_SPECIAL_TOKENS,
        image_tag_type=REFERENCE_TOKENIZER_IMAGE_TAG_TYPE,
        # The production launch script passes --tokenizer-keep-history-thinking
        # (line 451). Without it the constructor default (False) lets the chat
        # template's own default (True) truncate history thinking, which the
        # recipe never does. NeMo-RL hardcodes the same behaviour at
        # nemotron_tokenization.py:286.
        keep_history_thinking=True,
    )


class RecordingArgs(SimpleNamespace):
    """Megatron args namespace that records every attribute it could not supply.

    The reference encoder reads args from three places: arguments.py,
    TransformerConfig, and recipe-specific extras. Rather than guess the full
    set, anything missing returns None and is recorded, so a run always reports
    which arguments were defaulted instead of failing one at a time.
    """

    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__(**values)
        object.__setattr__(self, "_defaulted", set())

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        object.__getattribute__(self, "_defaulted").add(name)
        return None

    def defaulted(self) -> list[str]:
        return sorted(object.__getattribute__(self, "_defaulted"))


def harvest_transformer_config_defaults(reference_root: Path) -> dict[str, Any]:
    """Read TransformerConfig dataclass field defaults without importing it."""
    import ast

    path = reference_root / "megatron/core/transformer/transformer_config.py"
    defaults: dict[str, Any] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign) or statement.value is None:
                continue
            if not isinstance(statement.target, ast.Name):
                continue
            try:
                defaults[statement.target.id] = ast.literal_eval(statement.value)
            except ValueError:
                continue
    return defaults


def build_reference_encoder(args_path: Path, tokenizer: Any) -> Any:
    """Patch Megatron's globals, then import and construct the encoder.

    ``task_encoder`` binds ``get_args``/``get_tokenizer`` at import time, so the
    patch has to land before the module is imported.
    """
    import types

    args = RecordingArgs(json.loads(args_path.read_text()))
    training = sys.modules.get("megatron.training") or types.ModuleType(
        "megatron.training"
    )
    training.get_args = lambda: args
    training.get_tokenizer = lambda: tokenizer
    sys.modules["megatron.training"] = training

    from importlib import import_module

    module = import_module("data_loading.task_encoder")
    return module.MultiModalTaskEncoder(is_val=True), args


def build_nemo_rl_encoder(
    config_path: Path,
    model_path: Path,
    shared_tokenizer: Path | None = None,
    option_overrides: dict[str, Any] | None = None,
) -> Any:
    """Build the configured NeMo-RL task encoder from the SFT recipe.

    ``shared_tokenizer`` replaces ``processor.tokenizer`` so both stacks
    tokenize with the same vocabulary and the same chat template. The image
    processing config still comes from ``model_path``.
    """
    from transformers import AutoProcessor

    from nemo_rl.data.energon.config import EnergonLoaderConfig
    from nemo_rl.data.energon.multimodal.registry import (
        COOKER_REGISTRY,
        TASK_ENCODER_REGISTRY,
    )
    from nemo_rl.data.energon.multimodal.task_encoders import build_processor_adapter
    from nemo_rl.utils.config import load_config

    from megatron.energon import Cooker

    config = load_config(str(config_path))
    data = config["data"]
    loader_config = EnergonLoaderConfig(**data["energon"])
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    if shared_tokenizer is not None:
        from transformers import AutoTokenizer

        processor.tokenizer = AutoTokenizer.from_pretrained(
            str(shared_tokenizer), trust_remote_code=True
        )
    adapter = build_processor_adapter(
        processor_adapter=loader_config.processor_adapter,
        processor=processor,
        max_sequence_length=int(config["policy"]["max_total_sequence_length"]),
        add_bos=data.get("add_bos", True),
        add_eos=data.get("add_eos", True),
        add_generation_prompt=data.get("add_generation_prompt", False),
    )
    encoder_type = TASK_ENCODER_REGISTRY.resolve(loader_config.task_encoder.name)
    options: dict[str, Any] = {}
    if loader_config.task_encoder.name == "nemotron_multimodal":
        options = loader_config.task_encoder.options.model_dump()
    if option_overrides:
        unknown = set(option_overrides) - set(options)
        if unknown:
            raise ValueError(
                f"Unknown task encoder options: {sorted(unknown)}. "
                f"Known: {sorted(options)}"
            )
        for key, raw in option_overrides.items():
            current = options[key]
            if isinstance(current, bool):
                options[key] = raw.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                options[key] = int(raw)
            elif isinstance(current, float):
                options[key] = float(raw)
            elif raw.lower() in ("none", "null"):
                options[key] = None
            elif current is None and raw.isdigit():
                options[key] = int(raw)
            else:
                options[key] = raw
    return encoder_type(
        adapter=adapter,
        cooker_functions=[
            Cooker(
                COOKER_REGISTRY.resolve(cooker.name),
                has_subflavors=cooker.has_subflavors,
            )
            for cooker in loader_config.cookers
        ],
        packing_hooks=None,
        include_source_ids=True,
        **options,
    )


def expand_reference_final(
    encoded: Any, *, encoder: Any, image_token_id: int, temporal_patch_size: int
) -> tuple[list[int], list[int]]:
    """Expand the reference's -200 sentinels into the model-visible sequence.

    The reference never expands in the dataloader; LLaVA replaces each
    DEFAULT_IMAGE_TOKEN_INDEX with that media item's embeddings. Grouping uses
    the reference's own _group_video_frame_params_into_tubelets, because one
    IMAGE_TOKEN is emitted per tubelet while `images` stores ungrouped params.
    """
    params = list(encoded.images or [])
    grouped = (
        encoder._group_video_frame_params_into_tubelets(
            [p.media for p in params], params, temporal_patch_size
        )
        if temporal_patch_size > 1
        else params
    )
    widths = [p.num_embeddings for p in grouped]

    tokens = _as_list(encoded.tokens)
    labels = _as_list(encoded.labels)
    sentinels = [i for i, t in enumerate(tokens) if t < 0]
    if len(sentinels) != len(widths):
        raise ValueError(
            f"reference has {len(sentinels)} image sentinels for {len(widths)} "
            "grouped visual params"
        )

    out_tokens: list[int] = []
    out_labels: list[int] = []
    start = 0
    for position, width in zip(sentinels, widths, strict=True):
        out_tokens.extend(tokens[start:position])
        out_labels.extend(labels[start:position])
        out_tokens.extend([image_token_id] * width)
        out_labels.extend([labels[position]] * width)
        start = position + 1
    out_tokens.extend(tokens[start:])
    out_labels.extend(labels[start:])
    return out_tokens, out_labels


def expand_nemo_rl_final(encoded: Any, *, image_token_id: int) -> tuple[list[int], list[int]]:
    """Run NeMo-RL's own expansion without loading pixels.

    postencode() loads media, but the widths come from the saved plan
    (nemotron_multimodal.py:831), so the expansion is reproducible from the plan
    alone.
    """
    from copy import deepcopy

    from nemo_rl.data.energon.multimodal.task_encoders.nemotron_visual import (
        _expand_visual_placeholders,
    )

    message_log = deepcopy(encoded.message_log)
    occurrences = [
        (plan.message_index, plan.embedding_widths)
        for plan in getattr(encoded, "visual_plans", ()) or ()
    ]
    _expand_visual_placeholders(
        message_log, occurrences, image_token_id=image_token_id
    )
    tokens: list[int] = []
    mask: list[int] = []
    for message in message_log:
        tokens.extend(_as_list(message["token_ids"]))
        mask.extend(int(v) for v in _as_list(message["token_loss_mask"]))
    return tokens, mask


def _as_list(tensor: Any) -> list[Any]:
    if tensor is None:
        return []
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return list(tensor)


def normalize_reference_encoded(
    encoded: Any,
    *,
    ignore_index: int,
    sentinel_map: dict[int, int],
    encoder: Any,
    image_token_id: int,
    temporal_patch_size: int,
) -> dict[str, Any]:
    # The reference carries DEFAULT_IMAGE_TOKEN_INDEX (-200) and
    # DEFAULT_SOUND_TOKEN_INDEX (-300) as placeholders and expands them after
    # pre-encoding; NeMo-RL writes the real vocabulary ids. Map the sentinels so
    # the two sequences are comparable.
    tokens = [sentinel_map.get(t, t) for t in _as_list(encoded.tokens)]
    labels = _as_list(encoded.labels)
    mask = [0 if label == ignore_index else 1 for label in labels]
    # The reference stores UNGROUPED per-frame params (task_encoder.py:1031), so
    # a 20-frame video yields 20 entries. Compare per-frame embedding widths so
    # this lines up with NeMo-RL's per-plan embedding_widths.
    sizes = [params.num_embeddings for params in encoded.images or []]
    final_tokens, final_labels = expand_reference_final(
        encoded,
        encoder=encoder,
        image_token_id=image_token_id,
        temporal_patch_size=temporal_patch_size,
    )
    final_mask = [0 if label == ignore_index else 1 for label in final_labels]
    return {
        "final_tokens": final_tokens,
        "final_loss_mask": final_mask,
        "final_length": len(final_tokens),
        "tokens": tokens,
        "loss_mask": mask,
        "trainable_tokens": sum(mask),
        "image_count": len(sizes),
        "image_sizes": sizes,
        "num_frames": list(encoded.num_frames or []),
        "total_len": int(encoded.total_len),
    }


def normalize_nemo_rl_encoded(encoded: Any, *, image_token_id: int) -> dict[str, Any]:
    tokens: list[int] = []
    mask: list[int] = []
    for message in encoded.message_log:
        tokens.extend(_as_list(message.get("token_ids")))
        loss = message.get("token_loss_mask")
        if loss is None:
            mask.extend([0] * len(_as_list(message.get("token_ids"))))
        else:
            mask.extend(int(v) for v in _as_list(loss))
    plans = list(getattr(encoded, "visual_plans", ()) or ())
    # NeMo-RL keeps one plan per media item with a width per frame. Flatten to a
    # per-frame list so it matches the reference's ungrouped params.
    sizes = [width for plan in plans for width in plan.embedding_widths]
    frames = [len(plan.patch_sizes) for plan in plans]
    final_tokens, final_mask = expand_nemo_rl_final(
        encoded, image_token_id=image_token_id
    )
    return {
        "final_tokens": final_tokens,
        "final_loss_mask": final_mask,
        "final_length": len(final_tokens),
        "tokens": tokens,
        "loss_mask": mask,
        "trainable_tokens": sum(mask),
        "image_count": len(sizes),
        "image_sizes": sizes,
        "num_frames": frames,
        # packing_cost matches the reference formula exactly:
        #   len(input_ids) + image_embeddings - num_images   (task_encoder.py:2002)
        #   length         + visual_embeddings - placeholders (nemotron_visual.py:1119)
        # encoded.length is the compact text count and has no counterpart.
        "total_len": int(encoded.packing_cost),
    }


def _tensor_list(value: Any) -> list[Any]:
    """Flatten a tensor, a PackedTensor, or a list of either into tensors."""
    import torch

    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    packed = getattr(value, "tensors", None)
    if packed is not None:
        return [t for t in packed if t is not None]
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_tensor_list(item))
        return out
    return []


def normalize_reference_post(encoded: Any) -> dict[str, Any]:
    """Reference post-encoding: PackedTaskSample.

    ``tokens`` stays compact here -- the reference never expands -- so the
    comparable length is ``max_length``, which records the expanded size.
    """
    pixels = _tensor_list(getattr(encoded, "imgs", None))
    return {
        "post_length": int(encoded.max_length),
        "post_pixel_shapes": [tuple(t.shape) for t in pixels],
        "post_pixel_values": pixels,
        "post_num_tiles": list(getattr(encoded, "num_tiles", []) or []),
        "post_num_frames": list(getattr(encoded, "num_frames", []) or []),
    }


def normalize_nemo_rl_post(encoded: Any) -> dict[str, Any]:
    """NeMo-RL post-encoding: message_log carries the expanded stream."""
    length = sum(len(message["token_ids"]) for message in encoded.message_log)
    pixels: list[Any] = []
    sizes: list[Any] = []
    frames: list[Any] = []
    for message in encoded.message_log:
        pixels.extend(_tensor_list(message.get("pixel_values")))
        sizes.extend(_tensor_list(message.get("imgs_sizes")))
        frames.extend(_tensor_list(message.get("num_frames")))
    return {
        "post_length": length,
        "post_pixel_shapes": [tuple(t.shape) for t in pixels],
        "post_pixel_values": pixels,
        # imgs_sizes rows are (h, w) per tile; the reference reports a tile
        # count per media item, so compare counts rather than the raw rows.
        "post_num_tiles": [int(t.shape[0]) for t in sizes],
        "post_num_frames": [int(v) for t in frames for v in t.flatten().tolist()],
    }


def _pixels_match(left: list[Any], right: list[Any]) -> bool:
    import torch

    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a.shape != b.shape:
            return False
        if not torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-4):
            return False
    return True


def classify_encode_diff(ref: dict[str, Any], nrl: dict[str, Any]) -> list[str]:
    found = []
    for field in ENCODE_FIELDS:
        if field not in ref and field not in nrl:
            continue
        if field == "post_pixel_values":
            if not _pixels_match(ref.get(field, []), nrl.get(field, [])):
                found.append(field)
        elif ref.get(field) != nrl.get(field):
            found.append(field)
    return found


def _first_token_mismatch(ref: list[int], nrl: list[int]) -> dict[str, Any]:
    for index, (left, right) in enumerate(zip(ref, nrl)):
        if left != right:
            return {
                "index": index,
                "reference": ref[max(0, index - 4) : index + 5],
                "nemo_rl": nrl[max(0, index - 4) : index + 5],
            }
    return {"length_only": {"reference": len(ref), "nemo_rl": len(nrl)}}


def run(args: argparse.Namespace) -> int:
    if args.emit_reference_args_template:
        Path(args.emit_reference_args_template).write_text(
            json.dumps(REFERENCE_ARGS_TEMPLATE, indent=2)
        )
        print(f"wrote {args.emit_reference_args_template}")
        return 0

    reference_root = Path(args.reference_root).expanduser().resolve()
    sys.path.insert(0, str(reference_root))
    # image_processing.py:18 uses an absolute `from data_loading...` import, so
    # examples/multimodal has to be importable as a top-level package root too.
    sys.path.insert(0, str(reference_root / "examples" / "multimodal"))
    try:
        ref_modules = [cc.import_module(name) for name in REFERENCE_MODULES]
    except ImportError as error:
        print(f"cannot import the reference cookers: {error}")
        return 2
    nrl_modules = [cc.import_module(name) for name in cc.NEMO_RL_MODULES]

    def lookup(modules, name: str):
        for module in modules:
            found = getattr(module, name, None)
            if found is not None:
                return found
        return None

    logging.disable(logging.WARNING)

    # Energon already runs single-worker (WorkerConfig num_workers=0 below), but
    # torch/PIL/av still spawn intra-op threads. Cap them so a long sweep does
    # not saturate a shared login node.
    import torch

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        # Only settable before any parallel work has started; ignore if late.
        pass

    # Both encoders reach for WorkerConfig.active_worker_config through Energon's
    # WorkerRng, and for the sample-index stack that seeds it. Outside a real
    # loader nothing sets either, so activate a single-rank, zero-worker config.
    # Pushing the same index around both encode calls gives them the same seed
    # for the same row, which matters because data augmentation draws from it.
    from megatron.energon import WorkerConfig

    WorkerConfig.active_worker_config = None
    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
    worker_config.worker_activate(0)

    try:
        shared_tokenizer = (
            Path(args.shared_tokenizer).expanduser() if args.shared_tokenizer else None
        )
        overrides: dict[str, Any] = {}
        for item in args.nemo_rl_option or []:
            key, _, value = item.partition("=")
            overrides[key.strip()] = value.strip()
        nrl_encoder = build_nemo_rl_encoder(
            Path(args.config).expanduser(),
            Path(args.model_path).expanduser(),
            shared_tokenizer,
            overrides,
        )
    except ImportError as error:
        print(
            f"cannot build the NeMo-RL task encoder: {error}\n"
            "Needs transformers plus the energon extra: "
            "uv run --locked --extra energon --extra mcore tools/compare_task_encoders.py ..."
        )
        return 2
    except (OSError, ValueError, KeyError) as error:
        print(
            f"cannot build the NeMo-RL task encoder: {type(error).__name__}: {error}\n"
            f"Check --config {args.config} and --model-path {args.model_path}."
        )
        return 2

    ref_encoder = None
    ignore_index = -100
    sentinel_map: dict[int, int] = {}
    image_token_id = -1
    if not args.skip_reference:
        if not args.reference_args:
            print(
                "--reference-args is required unless --skip-reference is set. "
                "Use --emit-reference-args-template to write a skeleton."
            )
            return 2
        try:
            install_megatron_core_stubs(reference_root, verbose=not args.quiet)
            from megatron.core.models.multimodal.llava_model import IGNORE_INDEX

            ref_encoder, reference_args = build_reference_encoder(
                Path(args.reference_args).expanduser(),
                build_reference_tokenizer(
                    reference_root,
                    shared_tokenizer
                    or Path(args.tokenizer_path).expanduser(),
                ),
            )
            ignore_index = IGNORE_INDEX
            llava = sys.modules["megatron.core.models.multimodal.llava_model"]
            from transformers import AutoTokenizer

            nrl_tokenizer = AutoTokenizer.from_pretrained(
                str(shared_tokenizer or Path(args.model_path).expanduser()),
                trust_remote_code=True,
            )
            image_token_id = nrl_tokenizer.convert_tokens_to_ids(llava.IMAGE_TOKEN)
            sentinel_map = {
                llava.DEFAULT_IMAGE_TOKEN_INDEX: image_token_id,
                llava.DEFAULT_SOUND_TOKEN_INDEX: nrl_tokenizer.convert_tokens_to_ids(
                    llava.SOUND_TOKEN
                ),
            }
        except ImportError as error:
            print(
                f"cannot build the reference task encoder: {error}\n"
                "Needs transformers. megatron-core is stubbed from the reference "
                "source, so it should not be required; re-run with --skip-reference "
                "to compare cookers only."
            )
            return 2
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            print(
                f"cannot build the reference task encoder: "
                f"{type(error).__name__}: {error}\n"
                f"The args skeleton in --reference-args ({args.reference_args}) is "
                "probably missing a field the constructor reads. Add it and re-run."
            )
            return 2

    leaves = cc.load_leaves(Path(args.subset).expanduser(), args.split)
    if args.leaf:
        leaves = [leaf for leaf in leaves if args.leaf in leaf["name"]]

    started = time.monotonic()
    rows_done = 0
    leaf_index = 0
    leaf_total = len(leaves)
    if args.progress:
        print(
            f"{'#':>3}/{leaf_total:<3} {'leaf':<52} {'rows':>6} {'ident':>6} "
            f"{'diff':>6} {'err':>5} {'rows/s':>7} {'eta':>8}",
            file=sys.stderr,
            flush=True,
        )

    per_leaf: dict[str, Counter] = defaultdict(Counter)
    encode_fields: Counter = Counter()
    cook_fields: Counter = Counter()
    examples: list[dict[str, Any]] = []

    for leaf in leaves:
        pair = cc.COOK_PAIRS.get(leaf["cook"])
        if pair is None:
            per_leaf[leaf["name"]]["skipped_no_cooker_pair"] += 1
            continue
        ref_name, nrl_name, accepts = pair
        ref_cook = lookup(ref_modules, ref_name)
        nrl_cook = lookup(nrl_modules, nrl_name)
        if ref_cook is None or nrl_cook is None:
            per_leaf[leaf["name"]]["skipped_no_cooker_pair"] += 1
            continue

        stats = per_leaf[leaf["name"]]
        media_store = (
            None if leaf["media_root"] is None else cc.StubStore(leaf["media_root"])
        )
        primary_store = cc.StubStore(leaf["jsonl"].parent)
        aux_stores = {key: cc.StubStore(path) for key, path in leaf["aux"].items()}

        for index, payload in cc.iter_rows(leaf["jsonl"], args.limit):
            if index is None:
                stats["source_unreadable"] += payload
                break
            stats["rows"] += 1

            kwargs: dict[str, Any] = {}
            if "media_source" in accepts:
                kwargs["media_source"] = media_store
            if "aux" in accepts:
                kwargs.update(aux_stores)
            if "primary" in accepts:
                kwargs["primary"] = primary_store

            sink = io.StringIO()
            ref_post = nrl_post = None
            ref_cooked = nrl_cooked = None
            ref_error = nrl_error = None
            stage = "cook"
            with contextlib.redirect_stdout(sink):
                try:
                    ref_cooked = ref_cook(
                        cc.build_sample(leaf, index, copy.deepcopy(payload)),
                        cache=cc.StubCache(),
                        **kwargs,
                    )
                except Exception as error:  # noqa: BLE001 - parity comparison
                    ref_error = f"{type(error).__name__}: {error}"
                try:
                    nrl_cooked = nrl_cook(
                        cc.build_sample(leaf, index, copy.deepcopy(payload)),
                        cache=cc.StubCache(),
                        **kwargs,
                    )
                except Exception as error:  # noqa: BLE001 - parity comparison
                    nrl_error = f"{type(error).__name__}: {error}"

                if ref_cooked is not None and nrl_cooked is not None:
                    stage = "encode"
                    worker_config.worker_push_sample_index(index)
                    try:
                        if ref_encoder is not None:
                            try:
                                ref_cooked = ref_encoder.preencode_sample(ref_cooked)
                            except Exception as error:  # noqa: BLE001
                                ref_error = f"{type(error).__name__}: {error}"
                                ref_cooked = None
                        try:
                            nrl_cooked = nrl_encoder.preencode_sample(nrl_cooked)
                        except Exception as error:  # noqa: BLE001
                            nrl_error = f"{type(error).__name__}: {error}"
                            nrl_cooked = None
                        if args.postencode and not (ref_error or nrl_error):
                            try:
                                ref_post = ref_encoder.postencode_sample(ref_cooked)
                            except Exception as error:  # noqa: BLE001
                                ref_error = f"postencode: {type(error).__name__}: {error}"
                            try:
                                nrl_post = nrl_encoder.postencode_sample(nrl_cooked)
                            except Exception as error:  # noqa: BLE001
                                nrl_error = f"postencode: {type(error).__name__}: {error}"
                    finally:
                        worker_config.worker_pop_sample_index()

            def record(kind: str, detail: dict[str, Any]) -> None:
                stats[kind] += 1
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "leaf": leaf["name"],
                            "row": index,
                            "stage": stage,
                            "kind": kind,
                            **detail,
                        }
                    )

            if ref_error and nrl_error:
                record(
                    f"both_raised_{stage}",
                    {"reference_error": ref_error[:300], "nemo_rl_error": nrl_error[:300]},
                )
                continue
            if ref_error:
                record(f"only_reference_raised_{stage}", {"reference_error": ref_error[:300]})
                continue
            if nrl_error:
                record(f"only_nemo_rl_raised_{stage}", {"nemo_rl_error": nrl_error[:300]})
                continue

            if args.skip_reference:
                stats["nemo_rl_ok"] += 1
                continue

            ref_norm = normalize_reference_encoded(
                ref_cooked,
                ignore_index=ignore_index,
                sentinel_map=sentinel_map,
                encoder=ref_encoder,
                image_token_id=image_token_id,
                temporal_patch_size=reference_args.video_temporal_patch_size or 1,
            )
            nrl_norm = normalize_nemo_rl_encoded(
                nrl_cooked, image_token_id=image_token_id
            )
            if ref_post is not None and nrl_post is not None:
                ref_norm.update(normalize_reference_post(ref_post))
                nrl_norm.update(normalize_nemo_rl_post(nrl_post))
                stats["postencode_compared"] += 1
                if ref_norm["post_pixel_values"]:
                    stats["postencode_rows_with_pixels"] += 1
            elif args.postencode:
                stats["postencode_missing"] += 1
            found = classify_encode_diff(ref_norm, nrl_norm)
            if not found:
                stats["identical"] += 1
                continue
            for field in found:
                encode_fields[field] += 1
            detail: dict[str, Any] = {"fields": found}
            for key in ("tokens", "final_tokens", "final_loss_mask"):
                if key in found:
                    detail[key] = _first_token_mismatch(
                        ref_norm[key], nrl_norm[key]
                    )
            for field in found:
                if field in ("tokens", "loss_mask", "final_tokens", "final_loss_mask"):
                    continue
                if field == "post_pixel_values":
                    detail[field] = {
                        "reference_shapes": ref_norm.get("post_pixel_shapes"),
                        "nemo_rl_shapes": nrl_norm.get("post_pixel_shapes"),
                    }
                    continue
                detail[field] = {
                    "reference": ref_norm[field],
                    "nemo_rl": nrl_norm[field],
                }
            record("differ", detail)

        # Streaming progress: one line per finished leaf, on stderr so it stays
        # visible when stdout is redirected to a file.
        if args.progress:
            leaf_index += 1
            rows_done += stats["rows"]
            elapsed = time.monotonic() - started
            rate = rows_done / elapsed if elapsed else 0.0
            eta = (elapsed / leaf_index) * (leaf_total - leaf_index)
            errors = sum(
                count for key, count in stats.items() if key.startswith("only_")
            )
            print(
                f"{leaf_index:>3}/{leaf_total:<3} {leaf['name'][:52]:<52} "
                f"{stats['rows']:>6} {stats['identical']:>6} {stats['differ']:>6} "
                f"{errors:>5} {rate:>7.1f} {eta / 60:>6.1f}m",
                file=sys.stderr,
                flush=True,
            )

    if ref_encoder is not None:
        missing = reference_args.defaulted()
        if missing:
            print(
                f"\nWARNING: {len(missing)} Megatron args were not supplied and "
                f"defaulted to None:\n  {', '.join(missing)}\n"
                "Add them to --reference-args if any of them changes encoding."
            )

    if args.progress:
        print(
            f"done: {rows_done} rows in {time.monotonic() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    report(per_leaf, encode_fields, cook_fields, examples, args)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "per_leaf": {n: dict(c) for n, c in per_leaf.items()},
                    "encode_fields": dict(encode_fields),
                    "examples": examples,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {args.json}")

    totals = Counter()
    for counter in per_leaf.values():
        totals.update(counter)
    bad = sum(
        count for key, count in totals.items() if key.startswith("only_") or key == "differ"
    )
    return 1 if bad else 0


def report(per_leaf, encode_fields, cook_fields, examples, args) -> None:
    columns = sorted({key for counter in per_leaf.values() for key in counter})
    width = max((len(name) for name in per_leaf), default=10)
    print("=" * (width + 14 * len(columns)))
    print(f"{'leaf':<{width}}  " + "  ".join(f"{c[:12]:>12}" for c in columns))
    print("=" * (width + 14 * len(columns)))
    totals = Counter()
    for name in sorted(per_leaf):
        counter = per_leaf[name]
        totals.update(counter)
        print(f"{name:<{width}}  " + "  ".join(f"{counter[c]:>12}" for c in columns))
    print("-" * (width + 14 * len(columns)))
    print(f"{'TOTAL':<{width}}  " + "  ".join(f"{totals[c]:>12}" for c in columns))

    print("\n--- encoder fields that disagree (rows affected) ---")
    if encode_fields:
        for field in ENCODE_FIELDS:
            if encode_fields[field]:
                print(f"  {field:<20} {encode_fields[field]}")
    else:
        print("  none")

    if args.skip_reference:
        print("\nNOTE: --skip-reference was set; the encoder diff did not run.")

    if examples:
        print("\n--- examples ---")
        for example in examples:
            print(json.dumps(example, indent=2, default=str)[:1600])
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", default="~/data/super-test-blend/cook_subset_v2/subset.yaml"
    )
    parser.add_argument("--reference-root", default="energon-megatron-lm")
    parser.add_argument("--split", default="train")
    parser.add_argument("--leaf", default=None, help="Substring filter on leaf name.")
    parser.add_argument("--limit", type=int, default=25, help="Rows per leaf; 0 = all.")
    parser.add_argument(
        "--config",
        default=(
            "examples/configs/sft_v2_tests/"
            "vlm_sft-nemotron-omni-30ba3b-4n8g-megatron-tp4etp4-super-test-blend.v1.yaml"
        ),
        help="SFT recipe supplying the NeMo-RL energon block.",
    )
    parser.add_argument(
        "--model-path",
        default="~/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        help="HF model directory with the tokenizer and processor config.",
    )
    parser.add_argument("--reference-args", default=None, help="JSON of Megatron args.")
    parser.add_argument(
        "--emit-reference-args-template",
        default=None,
        help="Write a starting args skeleton to this path and exit.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Run cookers for both sides but only the NeMo-RL encoder.",
    )
    parser.add_argument(
        "--nemo-rl-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a NeMo-RL task encoder option, e.g. "
            "--nemo-rl-option thinking_trace_format=ultra. Repeatable."
        ),
    )
    parser.add_argument(
        "--shared-tokenizer",
        default=None,
        help=(
            "Tokenizer directory used by BOTH stacks. Removes vocabulary and "
            "chat-template differences from the diff. Overrides --tokenizer-path "
            "on the reference side and replaces processor.tokenizer on the "
            "NeMo-RL side."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default="~/data/nano_v35_sft_v10_closethink_unmask_orig6k_vlm_tokenizer",
        help="Tokenizer the reference launch script points --tokenizer-model at.",
    )
    parser.add_argument(
        "--postencode",
        action="store_true",
        help=(
            "Also run postencode on both sides and compare pixel tensors, tile "
            "counts, frame counts, and the real expanded length. Decodes and "
            "transforms media on both sides, so it is much slower."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "torch intra-op threads. Default 1 to keep a shared login node "
            "responsive; raise it on a dedicated compute node."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Hide stub details.")
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Suppress the per-leaf streaming progress lines on stderr.",
    )
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--json", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
