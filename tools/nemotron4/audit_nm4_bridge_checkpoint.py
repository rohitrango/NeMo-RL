#!/usr/bin/env python3
"""Audit every tensor loaded from an NM4 Megatron-Bridge checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = REPO_ROOT / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MCORE_ROOT = BRIDGE_ROOT / "3rdparty/Megatron-LM"
for source_root in (REPO_ROOT, BRIDGE_ROOT / "src", MCORE_ROOT):
    sys.path.insert(0, str(source_root))


def _walk(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)
    else:
        yield value


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _tensor_stats(tensor, chunk_elements: int = 8 * 1024 * 1024) -> dict[str, Any]:
    import torch

    flat = tensor.detach().reshape(-1)
    finite_count = 0
    value_sum = 0.0
    value_min = math.inf
    value_max = -math.inf
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements]
        if chunk.is_floating_point() or chunk.is_complex():
            finite = torch.isfinite(chunk)
            finite_count += int(finite.sum().item())
            numeric = chunk.real if chunk.is_complex() else chunk
            numeric = numeric.float()
            if finite.any():
                finite_numeric = numeric[finite]
                value_sum += float(finite_numeric.sum().item())
                value_min = min(value_min, float(finite_numeric.min().item()))
                value_max = max(value_max, float(finite_numeric.max().item()))
        else:
            finite_count += chunk.numel()
            numeric = chunk.to(torch.float32)
            value_sum += float(numeric.sum().item())
            if chunk.numel():
                value_min = min(value_min, float(numeric.min().item()))
                value_max = max(value_max, float(numeric.max().item()))
    return {
        "numel": flat.numel(),
        "finite": finite_count,
        "nonfinite": flat.numel() - finite_count,
        "min": None if value_min == math.inf else value_min,
        "max": None if value_max == -math.inf else value_max,
        "sum": value_sum,
    }


def _describe_extra_state(value: Any) -> dict[str, Any]:
    import torch

    if value is None:
        return {"kind": "None"}
    if isinstance(value, torch.Tensor):
        result = {
            "kind": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
        if value.numel() <= 1024:
            result["stats"] = _tensor_stats(value)
        return result
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(map(str, value))}
    return {"kind": type(value).__name__, "repr": repr(value)[:500]}


def _audit_loaded_model(model, checkpoint: Path) -> dict[str, Any]:
    import torch
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.dist_checkpointing.state_dict_utils import load_preprocess
    from megatron.core.dist_checkpointing.utils import extract_sharded_base
    from megatron.core.utils import unwrap_model

    rank = torch.distributed.get_rank()
    core_model = unwrap_model(model)
    sharded = {"model": core_model.sharded_state_dict()}
    sharded, _, _ = load_preprocess(sharded)
    sharded, _ = extract_sharded_base(sharded)

    local_entries = []
    for entry in _walk(sharded):
        if not isinstance(entry, ShardedTensor):
            continue
        if entry.data is None:
            raise AssertionError(f"Expected loaded data for {entry.key}")
        stats = _tensor_stats(entry.data)
        local_entries.append(
            {
                "key": entry.key,
                "dtype": str(entry.dtype),
                "global_shape": list(entry.global_shape),
                "local_shape": list(entry.local_shape),
                "global_offset": list(entry.global_offset),
                "replica_id": _json_value(entry.replica_id),
                "flattened_range": None
                if entry.flattened_range is None
                else [entry.flattened_range.start, entry.flattened_range.stop],
                "stats": stats,
            }
        )

    full_state = core_model.state_dict(keep_vars=True)
    extra_entries = []
    for key, value in full_state.items():
        if "_extra_state" not in key:
            continue
        module_path = key.removesuffix("._extra_state")
        try:
            module_type = type(core_model.get_submodule(module_path)).__name__
        except AttributeError:
            module_type = "<unresolved>"
        extra_entries.append(
            {
                "key": key,
                "module_type": module_type,
                "value": _describe_extra_state(value),
            }
        )

    gathered_entries = [None] * torch.distributed.get_world_size()
    gathered_extras = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered_entries, local_entries)
    torch.distributed.all_gather_object(gathered_extras, extra_entries)
    if rank != 0:
        return {}

    checkpoint_metadata = dist_checkpointing.load_tensors_metadata(str(checkpoint))
    checkpoint_info = {
        key: {
            "dtype": str(value.dtype),
            "global_shape": list(value.global_shape),
            "numel": math.prod(value.global_shape),
        }
        for key, value in checkpoint_metadata.items()
    }
    expected_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for process_entries in gathered_entries:
        for entry in process_entries:
            expected_by_key[entry["key"]].append(entry)

    checkpoint_keys = set(checkpoint_info)
    expected_keys = set(expected_by_key)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    shape_mismatches = []
    dtype_mismatches = []
    nonfinite = []
    for key, entries in expected_by_key.items():
        metadata = checkpoint_info.get(key)
        if metadata is not None:
            for entry in entries:
                if entry["global_shape"] != metadata["global_shape"]:
                    shape_mismatches.append(
                        {
                            "key": key,
                            "expected": entry["global_shape"],
                            "checkpoint": metadata["global_shape"],
                        }
                    )
                if entry["dtype"] != metadata["dtype"]:
                    dtype_mismatches.append(
                        {
                            "key": key,
                            "expected": entry["dtype"],
                            "checkpoint": metadata["dtype"],
                        }
                    )
        for entry in entries:
            if entry["stats"]["nonfinite"]:
                nonfinite.append(
                    {
                        "key": key,
                        "global_offset": entry["global_offset"],
                        "count": entry["stats"]["nonfinite"],
                    }
                )

    unique_extras: dict[str, dict[str, Any]] = {}
    extra_ranks: dict[str, list[int]] = defaultdict(list)
    for process_rank, process_entries in enumerate(gathered_extras):
        for entry in process_entries:
            unique_extras.setdefault(entry["key"], entry)
            extra_ranks[entry["key"]].append(process_rank)
    for key, entry in unique_extras.items():
        entry["ranks"] = extra_ranks[key]

    digest = hashlib.sha256()
    for key in sorted(expected_by_key):
        for entry in sorted(
            expected_by_key[key],
            key=lambda item: (item["global_offset"], str(item["replica_id"])),
        ):
            digest.update(
                json.dumps(
                    {
                        "key": key,
                        "dtype": entry["dtype"],
                        "global_shape": entry["global_shape"],
                        "global_offset": entry["global_offset"],
                        "stats": entry["stats"],
                    },
                    sort_keys=True,
                ).encode()
            )

    checkpoint_extra_keys = sorted(key for key in checkpoint_keys if "_extra_state" in key)
    return {
        "checkpoint": str(checkpoint),
        "world_size": torch.distributed.get_world_size(),
        "checkpoint_tensor_count": len(checkpoint_info),
        "checkpoint_global_numel": sum(item["numel"] for item in checkpoint_info.values()),
        "checkpoint_dtype_counts": dict(
            Counter(item["dtype"] for item in checkpoint_info.values())
        ),
        "expected_tensor_count": len(expected_by_key),
        "loaded_local_shard_count": sum(len(items) for items in gathered_entries),
        "loaded_local_numel": sum(
            entry["stats"]["numel"]
            for process_entries in gathered_entries
            for entry in process_entries
        ),
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "nonfinite_loaded_shards": nonfinite,
        "loaded_stats_sha256": digest.hexdigest(),
        "checkpoint_extra_state_keys": checkpoint_extra_keys,
        "model_extra_state_count": len(unique_extras),
        "model_extra_state_type_counts": dict(
            Counter(entry["module_type"] for entry in unique_extras.values())
        ),
        "model_extra_state_value_counts": dict(
            Counter(entry["value"]["kind"] for entry in unique_extras.values())
        ),
        "model_extra_state_entries": [unique_extras[key] for key in sorted(unique_extras)],
    }


def run(
    config_path: Path,
    checkpoint: Path,
    output: Path,
    *,
    preserve_checkpoint_router: bool,
) -> None:
    import torch
    from omegaconf import OmegaConf

    from nemo_rl.models.megatron.setup import (
        destroy_parallel_state,
        handle_model_import,
        setup_distributed,
        setup_model_and_optimizer,
        validate_and_set_config,
        validate_megatron_config,
        validate_model_paths,
    )
    from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    checkpoint = checkpoint.resolve()
    register_omegaconf_resolvers()
    resolved = OmegaConf.to_container(load_config(config_path.resolve()), resolve=True)
    policy = resolved["policy"]
    policy["pretrained_checkpoint"] = {
        "path": str(checkpoint),
        "format": "megatron_bridge",
    }
    if preserve_checkpoint_router:
        policy["megatron_cfg"]["moe_router_load_balancing_type"] = "quantile_balancing"
        policy["megatron_cfg"]["moe_aux_loss_coeff"] = 0.0
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    apply_transformer_engine_patch()
    setup_distributed(policy)
    try:
        rank = torch.distributed.get_rank()
        hf_model_name, pretrained_path, checkpoint_exists = validate_model_paths(policy)
        handle_model_import(policy, hf_model_name, pretrained_path, checkpoint_exists)
        runtime = validate_and_set_config(
            policy,
            rank,
            hf_model_name,
            pretrained_path,
            str(checkpoint),
            None,
        )
        validate_megatron_config(runtime.megatron_cfg, policy)
        state = setup_model_and_optimizer(
            policy,
            runtime.megatron_cfg,
            load_optimizer=False,
            load_weights=True,
        )
        result = _audit_loaded_model(state.model, checkpoint)
        if rank == 0:
            result["config"] = str(config_path.resolve())
            result["preserve_checkpoint_router"] = preserve_checkpoint_router
            result["provider"] = yaml.safe_load(
                (checkpoint / "run_config.yaml").read_text()
            )["model"]["_target_"]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            failures = sum(
                len(result[name])
                for name in (
                    "missing_checkpoint_keys",
                    "unexpected_checkpoint_keys",
                    "shape_mismatches",
                    "dtype_mismatches",
                    "nonfinite_loaded_shards",
                )
            )
            print(f"AUDIT_RESULT={output}")
            print(f"AUDIT_FAILURES={failures}")
        torch.distributed.barrier()
    finally:
        destroy_parallel_state()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preserve-checkpoint-router",
        action="store_true",
        help="Keep quantile-balanced routing so checkpoint router buffers are instantiated",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.checkpoint,
        args.output.resolve(),
        preserve_checkpoint_router=args.preserve_checkpoint_router,
    )


if __name__ == "__main__":
    main()
