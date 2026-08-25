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

"""Build an Energon SFT loader and iterate encoded batches without training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.sft import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.energon import build_energon_sft_loader
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, PackedTensor):
        tensors = [tensor for tensor in value.tensors if tensor is not None]
        return {
            "type": "packed_tensor",
            "rows": len(value),
            "tensor_shapes": [list(tensor.shape) for tensor in tensors],
            "tensor_dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        }
    if isinstance(value, Mapping):
        return {
            "type": type(value).__name__,
            "keys": sorted(str(key) for key in value),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "item_types": sorted({type(item).__name__ for item in value}),
        }
    return {"type": type(value).__name__}


def _message_log_summary(batch: Mapping[str, Any]) -> dict[str, Any] | None:
    message_logs = batch.get("message_log")
    if message_logs is None:
        message_logs = batch.get("packed_message_log")
    if not isinstance(message_logs, Sequence) or isinstance(
        message_logs, (str, bytes, bytearray)
    ):
        return None

    message_counts: list[int] = []
    token_counts: list[int] = []
    tensor_fields: set[str] = set()
    for row in message_logs:
        conversations = row if "packed_message_log" in batch else [row]
        row_message_count = 0
        row_token_count = 0
        for conversation in conversations:
            if not isinstance(conversation, Sequence):
                continue
            row_message_count += len(conversation)
            for message in conversation:
                if not isinstance(message, Mapping):
                    continue
                token_ids = message.get("token_ids")
                if isinstance(token_ids, torch.Tensor):
                    row_token_count += token_ids.numel()
                tensor_fields.update(
                    str(key)
                    for key, value in message.items()
                    if isinstance(value, (torch.Tensor, PackedTensor))
                )
        message_counts.append(row_message_count)
        token_counts.append(row_token_count)

    return {
        "rows": len(message_logs),
        "messages_per_row": message_counts,
        "tokens_per_row": token_counts,
        "tensor_fields": sorted(tensor_fields),
    }


def _loss_token_spans(
    batch: Mapping[str, Any], tokenizer: Any
) -> list[dict[str, Any]]:
    message_logs = batch.get("message_log")
    if message_logs is None:
        message_logs = batch.get("packed_message_log")
    if not isinstance(message_logs, Sequence) or isinstance(
        message_logs, (str, bytes, bytearray)
    ):
        return []

    samples: list[dict[str, Any]] = []
    for row_index, row in enumerate(message_logs):
        conversations = row if "packed_message_log" in batch else [row]
        for conversation_index, conversation in enumerate(conversations):
            if not isinstance(conversation, Sequence):
                continue
            spans: list[dict[str, Any]] = []
            for message_index, message in enumerate(conversation):
                if not isinstance(message, Mapping):
                    continue
                token_ids = message.get("token_ids")
                token_loss_mask = message.get("token_loss_mask")
                if (
                    not isinstance(token_ids, torch.Tensor)
                    or not isinstance(token_loss_mask, torch.Tensor)
                    or token_ids.ndim != 1
                    or token_loss_mask.shape != token_ids.shape
                ):
                    continue

                positions = torch.where(token_loss_mask != 0)[0].tolist()
                if not positions:
                    continue
                starts = [positions[0]]
                ends: list[int] = []
                for previous, current in zip(positions, positions[1:], strict=False):
                    if current != previous + 1:
                        ends.append(previous + 1)
                        starts.append(current)
                ends.append(positions[-1] + 1)

                for start, end in zip(starts, ends, strict=True):
                    loss_token_ids = token_ids[start:end].detach().cpu().tolist()
                    spans.append(
                        {
                            "message": message_index,
                            "start": start,
                            "end": end,
                            "token_count": end - start,
                            "decoded": tokenizer.decode(
                                loss_token_ids,
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=False,
                            ),
                        }
                    )
            samples.append(
                {
                    "row": row_index,
                    "conversation": conversation_index,
                    "trainable_token_count": sum(
                        span["token_count"] for span in spans
                    ),
                    "spans": spans,
                }
            )
    return samples


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="NeMo-RL SFT YAML path")
    parser.add_argument(
        "--steps",
        type=_positive_int,
        default=10,
        help="Number of encoded batches to consume",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Diagnostic batch size per logical data rank",
    )
    parser.add_argument("--logical-rank", type=int, default=0)
    parser.add_argument(
        "--logical-world-size",
        type=_positive_int,
        default=1,
        help="Number of logical data shards",
    )
    parser.add_argument(
        "--decode-loss-tokens",
        action="store_true",
        help="Decode each contiguous token span selected by token_loss_mask",
    )
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    return args, overrides


def main() -> None:
    args, overrides = _parse_args()
    if not 0 <= args.logical_rank < args.logical_world_size:
        raise ValueError("logical rank must be within the logical data world")

    register_omegaconf_resolvers()
    raw_config = load_config(args.config)
    if overrides:
        raw_config = parse_hydra_overrides(raw_config, overrides)
    config = MasterConfig.model_validate(
        OmegaConf.to_container(raw_config, resolve=True)
    )
    if config.data["backend"] != "energon":
        raise ValueError("This script requires data.backend=energon.")
    max_sequence_length = config.data["max_input_seq_length"]
    if max_sequence_length is None:
        raise ValueError("Energon SFT requires data.max_input_seq_length.")

    print(f"Loading processor: {config.policy['tokenizer']['name']}")
    processor = get_tokenizer(config.policy["tokenizer"], get_processor=True)
    print(
        "Building Energon loader: "
        f"rank={args.logical_rank}/{args.logical_world_size}, "
        f"batch_size={args.batch_size}"
    )
    loader = build_energon_sft_loader(
        data_config=config.data,
        source=config.data["train"],
        processor=processor,
        batch_size=args.batch_size,
        max_sequence_length=max_sequence_length,
        split_role="train",
        logical_rank=args.logical_rank,
        logical_world_size=args.logical_world_size,
        placement_fingerprint="standalone_energon_iterator",
    )

    iterator = iter(loader)
    samples_seen = 0
    for step in range(args.steps):
        batch = next(iterator)
        if not batch:
            raise RuntimeError(f"Energon returned an empty batch at step {step}.")
        source_ids = batch.get("source_ids", [])
        samples_seen += len(source_ids) if source_ids else args.batch_size
        summary = {
            "step": step,
            "source_ids": source_ids,
            "message_log": _message_log_summary(batch),
            "fields": {str(key): _value_summary(value) for key, value in batch.items()},
        }
        if args.decode_loss_tokens:
            tokenizer = getattr(processor, "tokenizer", processor)
            summary["loss_token_samples"] = _loss_token_spans(batch, tokenizer)
        print(
            json.dumps(
                summary,
                default=str,
                indent=2 if args.decode_loss_tokens else None,
                sort_keys=True,
            )
        )

    print(
        f"Energon iteration passed: batches={args.steps}, "
        f"samples={samples_seen}, rank={args.logical_rank}/"
        f"{args.logical_world_size}"
    )


if __name__ == "__main__":
    main()
