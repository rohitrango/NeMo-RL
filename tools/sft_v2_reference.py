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

"""Capture and compare deterministic Energon SFT reference windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.sft import MasterConfig, prepare_sft_batch
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.energon import (
    build_energon_sft_loader,
    build_energon_sft_dataloaders,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    byte_view = value.reshape(-1).view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _value_record(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
            "hash": _tensor_hash(value),
        }
    if isinstance(value, PackedTensor):
        tensors = [tensor for tensor in value.tensors if tensor is not None]
        return {
            "kind": "packed_tensor",
            "rows": len(value),
            "tensor_shapes": [tuple(tensor.shape) for tensor in tensors],
            "tensor_hashes": [_tensor_hash(tensor) for tensor in tensors],
        }
    return {"kind": type(value).__name__, "value": value}


def _source_ids(batch: Mapping[str, Any]) -> list[str]:
    for key in ("source_ids", "sample_keys"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
    return []


def _prepare_batch(
    batch: Mapping[str, Any], *, config: MasterConfig, processor: Any
) -> Mapping[str, Any]:
    return prepare_sft_batch(
        batch,
        tokenizer=processor.tokenizer,
        only_unmask_final=config.sft.only_unmask_final,
        make_sequence_length_divisible_by=config.policy[
            "make_sequence_length_divisible_by"
        ],
    )


def _prepared_row_records(
    raw_batch: Mapping[str, Any], prepared: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return padding-independent records for every prepared sample."""
    input_ids = prepared["input_ids"]
    input_lengths = prepared["input_lengths"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise TypeError("Prepared input_ids must be a rank-2 tensor.")
    if not isinstance(input_lengths, torch.Tensor) or input_lengths.ndim != 1:
        raise TypeError("Prepared input_lengths must be a rank-1 tensor.")
    batch_size, padded_sequence_length = input_ids.shape
    source_ids = _source_ids(raw_batch)
    if len(source_ids) != batch_size:
        raise ValueError(
            f"Prepared batch has {batch_size} rows but {len(source_ids)} source IDs."
        )

    rows: list[dict[str, Any]] = []
    for row_index, source_id in enumerate(source_ids):
        sequence_length = int(input_lengths[row_index].item())
        values: dict[str, Any] = {}
        for name, value in prepared.items():
            if isinstance(value, torch.Tensor):
                row_value = value
                if value.ndim > 0 and value.shape[0] == batch_size:
                    row_value = value[row_index]
                    if value.ndim >= 2 and value.shape[1] == padded_sequence_length:
                        row_value = row_value[:sequence_length]
                values[name] = _value_record(row_value)
            elif isinstance(value, PackedTensor):
                if len(value) != batch_size:
                    raise ValueError(
                        f"Packed field {name!r} has {len(value)} rows; expected "
                        f"{batch_size}."
                    )
                values[name] = _value_record(value.slice([row_index]))
            else:
                values[name] = _value_record(value)
        rows.append({"source_id": source_id, "values": values})
    return rows


def _row_hashes(rows: list[dict[str, Any]]) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for row in rows
    ]


def _compare_dp_batches(
    config_path: str,
    output_path: str,
    steps: int,
    skip: int,
    logical_world_size: int,
    overrides: list[str],
) -> None:
    """Verify V1 global preparation over the exact V2 logical-DP shards."""
    if steps <= 0 or skip < 0 or logical_world_size <= 0:
        raise ValueError(
            "Steps and logical world size must be positive; skip must be non-negative."
        )
    register_omegaconf_resolvers()
    raw_config = load_config(config_path)
    if overrides:
        raw_config = parse_hydra_overrides(raw_config, overrides)
    config = MasterConfig.model_validate(
        OmegaConf.to_container(raw_config, resolve=True)
    )
    global_batch_size = config.policy["train_global_batch_size"]
    if global_batch_size % logical_world_size != 0:
        raise ValueError(
            f"Global batch {global_batch_size} is not divisible by logical DP size "
            f"{logical_world_size}."
        )
    if config.data["shuffle"]:
        raise ValueError(
            "Exact DP batch comparison requires data.shuffle=false so V1 and V2 "
            "consume the same ordered sample window."
        )

    processor = get_tokenizer(config.policy["tokenizer"], get_processor=True)
    max_sequence_length = config.data["max_input_seq_length"]
    if max_sequence_length is None:
        raise ValueError("Reference capture requires data.max_input_seq_length.")
    local_batch_size = global_batch_size // logical_world_size
    shard_loaders = [
        build_energon_sft_loader(
            data_config=config.data,
            source=config.data["train"],
            processor=processor,
            batch_size=local_batch_size,
            max_sequence_length=max_sequence_length,
            split_role="train",
            logical_rank=logical_rank,
            logical_world_size=logical_world_size,
            placement_fingerprint=f"sft_v2_reference_dp{logical_world_size}",
        )
        for logical_rank in range(logical_world_size)
    ]
    shard_iterators = [iter(loader) for loader in shard_loaders]
    for _ in range(skip):
        for iterator in shard_iterators:
            next(iterator)

    summaries: list[dict[str, Any]] = []
    for step in range(steps):
        shard_batches = [next(iterator) for iterator in shard_iterators]
        combined_batch = BatchedDataDict.from_batches(shard_batches)
        combined_prepared = _prepare_batch(
            deepcopy(combined_batch), config=config, processor=processor
        )
        v1_dp_batches = combined_prepared.shard_by_batch_size(
            logical_world_size, batch_size=global_batch_size
        )
        if isinstance(v1_dp_batches, tuple):
            raise AssertionError("Fixed-batch V1 sharding unexpectedly reordered data.")

        v1_rows: list[dict[str, Any]] = []
        v2_rows: list[dict[str, Any]] = []
        rank_source_ids: list[list[str]] = []
        for shard_batch, v1_dp_batch in zip(
            shard_batches, v1_dp_batches, strict=True
        ):
            v2_prepared = _prepare_batch(
                deepcopy(shard_batch), config=config, processor=processor
            )
            v1_rows.extend(_prepared_row_records(shard_batch, v1_dp_batch))
            v2_rows.extend(_prepared_row_records(shard_batch, v2_prepared))
            rank_source_ids.append(_source_ids(shard_batch))

        v1_source_ids = [row["source_id"] for row in v1_rows]
        v2_source_ids = [row["source_id"] for row in v2_rows]
        if v1_source_ids != v2_source_ids:
            raise AssertionError(
                f"Rank-combined V1 and DP={logical_world_size} V2 source order "
                f"differs at step {step}: V1={v1_source_ids}, V2={v2_source_ids}."
            )
        v1_hashes = _row_hashes(v1_rows)
        v2_hashes = _row_hashes(v2_rows)
        if v1_hashes != v2_hashes:
            raise AssertionError(
                f"Rank-combined V1 and DP={logical_world_size} V2 prepared "
                f"samples differ at step {step}."
            )
        summaries.append(
            {
                "step": step,
                "source_ids_by_logical_rank": rank_source_ids,
                "prepared_row_hashes": v1_hashes,
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "format_version": 1,
                "config": config_path,
                "overrides": overrides,
                "logical_world_size": logical_world_size,
                "global_batch_size": global_batch_size,
                "steps": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"SFT rank-combined batch match: {steps} steps, "
        f"DP={logical_world_size}, global batch={global_batch_size}"
    )


def _capture(
    config_path: str,
    output_path: str,
    steps: int,
    skip: int,
    overrides: list[str],
) -> None:
    if steps <= 0 or skip < 0:
        raise ValueError(
            "Reference steps must be positive and skip must be non-negative."
        )
    register_omegaconf_resolvers()
    raw_config = load_config(config_path)
    if overrides:
        raw_config = parse_hydra_overrides(raw_config, overrides)
    config = MasterConfig.model_validate(
        OmegaConf.to_container(raw_config, resolve=True)
    )
    processor = get_tokenizer(config.policy["tokenizer"], get_processor=True)
    max_sequence_length = config.data["max_input_seq_length"]
    if max_sequence_length is None:
        raise ValueError("Reference capture requires data.max_input_seq_length.")
    loader, _ = build_energon_sft_dataloaders(
        data_config=config.data,
        processor=processor,
        train_batch_size=config.policy["train_global_batch_size"],
        val_batch_size=config.sft.val_global_batch_size,
        max_sequence_length=max_sequence_length,
    )

    def build_identity_loader() -> Any:
        return build_energon_sft_loader(
            data_config=config.data,
            source=config.data["train"],
            processor=processor,
            batch_size=config.policy["train_global_batch_size"],
            max_sequence_length=max_sequence_length,
            split_role="train",
            logical_rank=0,
            logical_world_size=1,
            placement_fingerprint="sft_v2_reference_dp1",
        )

    identity_loader = build_identity_loader()
    records: list[dict[str, Any]] = []
    iterator = iter(loader)
    identity_iterator = iter(identity_loader)
    for _ in range(skip):
        next(iterator)
        next(identity_iterator)
    for _ in range(steps):
        raw_batch = next(iterator)
        identity_batch = next(identity_iterator)
        prepared = prepare_sft_batch(
            raw_batch,
            tokenizer=processor.tokenizer,
            only_unmask_final=config.sft.only_unmask_final,
            make_sequence_length_divisible_by=config.policy[
                "make_sequence_length_divisible_by"
            ],
        )
        identity_prepared = prepare_sft_batch(
            identity_batch,
            tokenizer=processor.tokenizer,
            only_unmask_final=config.sft.only_unmask_final,
            make_sequence_length_divisible_by=config.policy[
                "make_sequence_length_divisible_by"
            ],
        )
        prepared_values = {key: _value_record(value) for key, value in prepared.items()}
        identity_values = {
            key: _value_record(value) for key, value in identity_prepared.items()
        }
        if prepared_values != identity_values:
            raise AssertionError(
                "The DP=1 V2 loader does not match the V1 prepared batch."
            )
        records.append(
            {
                "source_ids": _source_ids(identity_batch),
                "fields": list(prepared.keys()),
                "values": prepared_values,
                "valid_tokens": int(
                    (prepared["sample_mask"].unsqueeze(-1) * prepared["token_mask"])
                    .sum()
                    .item()
                ),
            }
        )
    loader_state = loader.state_dict()
    identity_state = identity_loader.state_dict()
    resumed_loader, _ = build_energon_sft_dataloaders(
        data_config=config.data,
        processor=processor,
        train_batch_size=config.policy["train_global_batch_size"],
        val_batch_size=config.sft.val_global_batch_size,
        max_sequence_length=max_sequence_length,
    )
    resumed_loader.load_state_dict(loader_state)
    resumed_identity_loader = build_identity_loader()
    resumed_identity_loader.load_state_dict(identity_state)
    resumed_raw = next(iter(resumed_loader))
    resumed_identity = next(iter(resumed_identity_loader))
    resumed_prepared = prepare_sft_batch(
        resumed_raw,
        tokenizer=processor.tokenizer,
        only_unmask_final=config.sft.only_unmask_final,
        make_sequence_length_divisible_by=config.policy[
            "make_sequence_length_divisible_by"
        ],
    )
    resumed_identity_prepared = prepare_sft_batch(
        resumed_identity,
        tokenizer=processor.tokenizer,
        only_unmask_final=config.sft.only_unmask_final,
        make_sequence_length_divisible_by=config.policy[
            "make_sequence_length_divisible_by"
        ],
    )
    if {key: _value_record(value) for key, value in resumed_prepared.items()} != {
        key: _value_record(value) for key, value in resumed_identity_prepared.items()
    }:
        raise AssertionError("The resumed DP=1 V2 batch does not match V1.")
    payload = {
        "format_version": 1,
        "config": config_path,
        "overrides": overrides,
        "steps": records,
        "loader_state": loader_state,
        "resumed_next_source_ids": _source_ids(resumed_identity),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def _compare(reference_path: str, candidate_path: str) -> None:
    reference = torch.load(reference_path, weights_only=False)
    candidate = torch.load(candidate_path, weights_only=False)
    reference_steps = reference.get("steps")
    if isinstance(candidate, list) and all("copies" in step for step in candidate):
        if any(len(step["copies"]) != 1 for step in candidate):
            raise ValueError(
                "Exact V1 tensor comparison requires a DP=1 V2 measurement."
            )
        candidate_steps = [step["copies"][0] for step in candidate]
    else:
        candidate_steps = candidate.get("steps")
    if len(reference_steps) != len(candidate_steps):
        raise AssertionError(
            f"SFT windows have {len(reference_steps)} and {len(candidate_steps)} steps."
        )
    for index, (reference_step, candidate_step) in enumerate(
        zip(reference_steps, candidate_steps, strict=True)
    ):
        compared_fields = ("fields", "values", "valid_tokens")
        if any(
            reference_step[field] != candidate_step[field] for field in compared_fields
        ):
            raise AssertionError(f"SFT prepared batch differs at step {index}.")
        if reference_step["source_ids"] and (
            reference_step["source_ids"] != candidate_step["source_ids"]
        ):
            raise AssertionError(f"SFT source order differs at step {index}.")
    print(f"SFT reference match: {len(reference_steps)} ordered steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--steps", type=int, default=4)
    capture.add_argument("--skip", type=int, default=0)
    capture.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override to apply before capture; repeat for multiple values.",
    )
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare_dp = subparsers.add_parser(
        "compare-dp",
        help="Verify V1 global preparation over V2 logical-DP shard unions.",
    )
    compare_dp.add_argument("--config", required=True)
    compare_dp.add_argument("--output", required=True)
    compare_dp.add_argument("--steps", type=int, default=4)
    compare_dp.add_argument("--skip", type=int, default=0)
    compare_dp.add_argument("--logical-world-size", type=int, required=True)
    compare_dp.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override to apply before comparison; repeat as needed.",
    )
    args = parser.parse_args()
    if args.command == "capture":
        _capture(args.config, args.output, args.steps, args.skip, args.override)
    elif args.command == "compare":
        _compare(args.reference, args.candidate)
    else:
        _compare_dp_batches(
            args.config,
            args.output,
            args.steps,
            args.skip,
            args.logical_world_size,
            args.override,
        )


if __name__ == "__main__":
    main()
