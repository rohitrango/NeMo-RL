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

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.energon.multimodal.packing.base import (
    ENERGON_PACKED_SCHEMA_VERSION,
)
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    message_log_to_flat_messages,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _pad(tensor: torch.Tensor, length: int, value: int | float = 0) -> torch.Tensor:
    if tensor.shape[0] > length:
        raise ValueError("A source tensor exceeds its selected padded length.")
    if tensor.shape[0] == length:
        return tensor
    padding = torch.full(
        (length - tensor.shape[0], *tensor.shape[1:]),
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=0)


def _stack_rows(
    rows: list[torch.Tensor], *, width: int, pad_value: int | float = 0
) -> torch.Tensor:
    return torch.stack([_pad(row, width, pad_value) for row in rows])


def _align_message_tensor(
    message_log: list[dict[str, Any]],
    key: str,
    template: torch.Tensor,
) -> torch.Tensor:
    """Insert zeros for turns that do not emit a sequence-side tensor."""
    aligned: list[torch.Tensor] = []
    for message in message_log:
        tokens = message.get("token_ids")
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("Packed SFT messages require token tensors.")
        value = message.get(key)
        if value is None:
            value = torch.zeros(
                (tokens.shape[0], *template.shape[1:]),
                dtype=template.dtype,
                device=template.device,
            )
        if (
            not isinstance(value, torch.Tensor)
            or value.shape[0] != tokens.shape[0]
            or value.shape[1:] != template.shape[1:]
        ):
            raise ValueError(f"Packed side tensor {key!r} is not turn-aligned.")
        aligned.append(value)
    return torch.cat(aligned)


def prepare_energon_packed_batch(
    batch: Mapping[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase,
    only_unmask_final: bool,
) -> BatchedDataDict[Any]:
    """Build model tensors from Energon-selected physical SFT packs."""
    if tokenizer.pad_token_id is None:
        raise ValueError("Energon packed SFT requires a tokenizer pad token id.")
    if batch.get("packed_schema_version") != ENERGON_PACKED_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing Energon packed SFT schema version.")

    packed_logs = batch.get("packed_message_log")
    source_padded_lengths = batch.get("source_padded_lengths")
    source_ids = batch.get("source_ids")
    loss_multipliers = batch.get("source_loss_multipliers")
    pack_capacity = batch.get("pack_capacity")
    if not isinstance(packed_logs, list) or not packed_logs:
        raise ValueError("Energon packed SFT batches require physical pack rows.")
    if not isinstance(pack_capacity, int) or pack_capacity <= 0:
        raise ValueError("Energon packed SFT batches require a positive pack capacity.")
    if not all(
        isinstance(value, list) and len(value) == len(packed_logs)
        for value in (source_padded_lengths, source_ids, loss_multipliers)
    ):
        raise ValueError("Energon packed SFT source metadata does not match pack count.")

    token_rows: list[torch.Tensor] = []
    token_mask_rows: list[torch.Tensor] = []
    pack_lengths: list[int] = []
    cu_seqlens: list[torch.Tensor] = []
    cu_seqlens_padded: list[torch.Tensor] = []
    side_rows: dict[str, list[torch.Tensor | None]] = {}
    media_rows: dict[str, list[PackedTensor | None]] = {}

    for pack_index, source_logs_value in enumerate(packed_logs):
        if not isinstance(source_logs_value, list) or not source_logs_value:
            raise ValueError("Each Energon physical pack needs at least one source.")
        padded_lengths = source_padded_lengths[pack_index]
        ids = source_ids[pack_index]
        multipliers = loss_multipliers[pack_index]
        if not (
            isinstance(padded_lengths, list)
            and isinstance(ids, list)
            and isinstance(multipliers, list)
            and len(source_logs_value)
            == len(padded_lengths)
            == len(ids)
            == len(multipliers)
        ):
            raise ValueError("One Energon physical pack has inconsistent source metadata.")

        source_logs = deepcopy(source_logs_value)
        add_loss_mask_to_message_log(
            source_logs,
            roles_to_train_on=["assistant"],
            only_unmask_final=only_unmask_final,
        )
        flattened = [message_log_to_flat_messages(log) for log in source_logs]
        source_lengths: list[int] = []
        for source_index, (flat, padded_length, multiplier) in enumerate(
            zip(flattened, padded_lengths, multipliers)
        ):
            tokens = flat.get("token_ids")
            mask = flat.get("token_loss_mask")
            if not isinstance(tokens, torch.Tensor) or not isinstance(
                mask, torch.Tensor
            ):
                raise TypeError("Packed SFT sources require token and loss-mask tensors.")
            source_length = tokens.shape[0]
            if source_length <= 0 or padded_length < source_length:
                raise ValueError(
                    f"Packed source {ids[source_index]!r} has invalid padded length."
                )
            source_lengths.append(source_length)
            source_mask = mask * float(multiplier)
            source_mask[0] = 0
            flattened[source_index]["token_loss_mask"] = source_mask

        selected_length = sum(padded_lengths)
        if selected_length > pack_capacity:
            raise ValueError("Energon physical pack exceeds its configured capacity.")
        pack_lengths.append(pack_capacity)
        cu_seqlens.append(
            torch.tensor(
                [0, *torch.tensor(source_lengths).cumsum(0).tolist()],
                dtype=torch.int32,
            )
        )
        padded_boundaries = [0, *torch.tensor(padded_lengths).cumsum(0).tolist()]
        padded_boundaries[-1] = pack_capacity
        cu_seqlens_padded.append(torch.tensor(padded_boundaries, dtype=torch.int32))

        token_rows.append(
            torch.cat(
                [
                    _pad(flat["token_ids"], padded_length, tokenizer.pad_token_id)
                    for flat, padded_length in zip(flattened, padded_lengths)
                ]
            )
        )
        token_mask_rows.append(
            torch.cat(
                [
                    _pad(flat["token_loss_mask"], padded_length)
                    for flat, padded_length in zip(flattened, padded_lengths)
                ]
            )
        )

        pack_index = len(pack_lengths) - 1
        token_side_keys = {
            key
            for flat in flattened
            for key, value in flat.items()
            if key not in {"token_ids", "token_loss_mask"}
            and isinstance(value, torch.Tensor)
        }
        for key in token_side_keys:
            template = next(
                flat[key]
                for flat in flattened
                if isinstance(flat.get(key), torch.Tensor)
            )
            assert isinstance(template, torch.Tensor)
            source_values: list[torch.Tensor] = []
            for source_index, (flat, source_length, padded_length) in enumerate(
                zip(flattened, source_lengths, padded_lengths)
            ):
                value = flat.get(key)
                if value is None:
                    value = torch.zeros(
                        (source_length, *template.shape[1:]),
                        dtype=template.dtype,
                        device=template.device,
                    )
                elif isinstance(value, torch.Tensor) and value.shape[0] != source_length:
                    value = _align_message_tensor(
                        source_logs[source_index], key, template
                    )
                if not isinstance(value, torch.Tensor) or value.shape[0] != source_length:
                    raise ValueError(
                        f"Packed side tensor {key!r} is not token-aligned for "
                        f"source {ids[source_index]!r}."
                    )
                source_values.append(_pad(value, padded_length))
            side_rows.setdefault(key, [None] * pack_index).append(
                torch.cat(source_values)
            )
        for key in side_rows.keys() - token_side_keys:
            side_rows[key].append(None)

        media_keys = {
            key
            for flat in flattened
            for key, value in flat.items()
            if isinstance(value, PackedTensor)
        }
        for key in media_keys:
            values = [flat[key] for flat in flattened if key in flat]
            assert all(isinstance(value, PackedTensor) for value in values)
            merged = PackedTensor.merge_segments(values)
            materialized = merged.as_tensor()
            media_rows.setdefault(key, [None] * pack_index).append(
                PackedTensor(
                    [materialized],
                    merged.dim_to_pack,
                    pad_to_max_shape=merged.pad_to_max_shape,
                )
            )
        for key in media_rows.keys() - media_keys:
            media_rows[key].append(None)

    width = pack_capacity
    prepared = BatchedDataDict(
        {
            "input_ids": _stack_rows(
                token_rows, width=width, pad_value=tokenizer.pad_token_id
            ),
            "input_lengths": torch.tensor(pack_lengths, dtype=torch.int32),
            "token_mask": _stack_rows(token_mask_rows, width=width),
            "sample_mask": torch.tensor(
                [float(mask.bool().any()) for mask in token_mask_rows],
                dtype=torch.float32,
            ),
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_padded": cu_seqlens_padded,
            "source_ids": source_ids,
            "source_lengths": [
                (boundaries[1:] - boundaries[:-1]).tolist()
                for boundaries in cu_seqlens
            ],
            "pack_capacity": torch.full(
                (len(packed_logs),), pack_capacity, dtype=torch.int32
            ),
            "packed_schema_version": torch.full(
                (len(packed_logs),),
                ENERGON_PACKED_SCHEMA_VERSION,
                dtype=torch.int32,
            ),
        }
    )
    for key, rows in side_rows.items():
        template = next((row for row in rows if row is not None), None)
        assert template is not None
        aligned_rows = [
            (
                row
                if row is not None
                else torch.zeros(
                    (pack_lengths[row_index], *template.shape[1:]),
                    dtype=template.dtype,
                    device=template.device,
                )
            )
            for row_index, row in enumerate(rows)
        ]
        prepared[key] = _stack_rows(aligned_rows, width=width)
    for key, rows in media_rows.items():
        template = next((row for row in rows if row is not None), None)
        assert template is not None
        aligned = [
            row if row is not None else PackedTensor.empty_rows_like(template, 1)
            for row in rows
        ]
        prepared[key] = PackedTensor.flattened_concat(aligned)
    return prepared


__all__ = ["ENERGON_PACKED_SCHEMA_VERSION", "prepare_energon_packed_batch"]
