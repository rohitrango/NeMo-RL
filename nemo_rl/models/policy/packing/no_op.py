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
"""Validation boundary for batches that Energon has already packed."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.models.policy.packing.base import (
    Packer,
    PackingInput,
    PackingResult,
    PlacedPackingInput,
)

ENERGON_PACKING_META_KEY = "energon_packing"
ENERGON_PACKING_SCHEMA_VERSION = 1


def _require_int(value: Any, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return value


def _require_int_list(
    value: Any, *, name: str, length: int, minimum: int
) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list with {length} items.")
    return [
        _require_int(item, name=f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    ]


def _validate_boundaries(
    value: Any,
    *,
    rank: int,
    source_counts: list[int],
    pack_lengths: list[int],
) -> None:
    if not isinstance(value, list) or len(value) != len(pack_lengths):
        raise ValueError(
            f"DP rank {rank} boundaries must contain one item per physical pack."
        )
    for pack_index, boundary in enumerate(value):
        if not isinstance(boundary, Mapping):
            raise ValueError(
                f"DP rank {rank} boundaries[{pack_index}] must be a mapping."
            )
        boundary_length = source_counts[pack_index] + 1
        cu_seqlens = _require_int_list(
            boundary.get("cu_seqlens"),
            name=f"DP rank {rank} boundaries[{pack_index}].cu_seqlens",
            length=boundary_length,
            minimum=0,
        )
        cu_seqlens_padded = _require_int_list(
            boundary.get("cu_seqlens_padded"),
            name=f"DP rank {rank} boundaries[{pack_index}].cu_seqlens_padded",
            length=boundary_length,
            minimum=0,
        )
        if cu_seqlens[0] != 0 or cu_seqlens_padded[0] != 0:
            raise ValueError(
                f"DP rank {rank} pack {pack_index} boundaries must start at zero."
            )
        if any(left >= right for left, right in zip(cu_seqlens, cu_seqlens[1:])):
            raise ValueError(
                f"DP rank {rank} pack {pack_index} cu_seqlens must be strictly "
                "increasing."
            )
        if any(
            left >= right
            for left, right in zip(cu_seqlens_padded, cu_seqlens_padded[1:])
        ):
            raise ValueError(
                f"DP rank {rank} pack {pack_index} cu_seqlens_padded must be "
                "strictly increasing."
            )
        source_lengths = [
            right - left for left, right in zip(cu_seqlens, cu_seqlens[1:])
        ]
        padded_lengths = [
            right - left
            for left, right in zip(cu_seqlens_padded, cu_seqlens_padded[1:])
        ]
        if any(
            source_length > padded_length
            for source_length, padded_length in zip(source_lengths, padded_lengths)
        ):
            raise ValueError(
                f"DP rank {rank} pack {pack_index} has a source length larger "
                "than its padded boundary."
            )
        if cu_seqlens_padded[-1] != pack_lengths[pack_index]:
            raise ValueError(
                f"DP rank {rank} pack {pack_index} padded boundary must end at "
                f"pack length {pack_lengths[pack_index]}."
            )


def _validate_meta(meta: KVBatchMeta, *, rank: int) -> tuple[int, int]:
    packing_meta = meta.extra_info.get(ENERGON_PACKING_META_KEY)
    if not isinstance(packing_meta, Mapping):
        raise ValueError(
            f"DP rank {rank} is missing {ENERGON_PACKING_META_KEY!r} metadata."
        )
    schema_version = packing_meta.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != ENERGON_PACKING_SCHEMA_VERSION
    ):
        raise ValueError(
            f"DP rank {rank} has unsupported Energon packing schema version "
            f"{schema_version!r}; expected {ENERGON_PACKING_SCHEMA_VERSION}."
        )

    pack_count = _require_int(
        packing_meta.get("pack_count"),
        name=f"DP rank {rank} pack_count",
        minimum=1,
    )
    source_count = _require_int(
        packing_meta.get("source_count"),
        name=f"DP rank {rank} source_count",
        minimum=1,
    )
    if len(meta.sample_ids) != pack_count:
        raise ValueError(
            f"DP rank {rank} has {len(meta.sample_ids)} sample IDs for "
            f"pack_count={pack_count}."
        )
    if meta.sequence_lengths is None or len(meta.sequence_lengths) != pack_count:
        raise ValueError(
            f"DP rank {rank} sequence_lengths must contain one item per "
            "physical pack."
        )

    source_counts = _require_int_list(
        packing_meta.get("source_counts"),
        name=f"DP rank {rank} source_counts",
        length=pack_count,
        minimum=1,
    )
    if sum(source_counts) != source_count:
        raise ValueError(
            f"DP rank {rank} source_counts sum to {sum(source_counts)}, "
            f"expected source_count={source_count}."
        )
    pack_lengths = _require_int_list(
        packing_meta.get("pack_lengths"),
        name=f"DP rank {rank} pack_lengths",
        length=pack_count,
        minimum=1,
    )
    if pack_lengths != meta.sequence_lengths:
        raise ValueError(
            f"DP rank {rank} pack_lengths must match KVBatchMeta.sequence_lengths."
        )
    pack_capacity = _require_int(
        packing_meta.get("pack_capacity"),
        name=f"DP rank {rank} pack_capacity",
        minimum=1,
    )
    if any(pack_length > pack_capacity for pack_length in pack_lengths):
        raise ValueError(
            f"DP rank {rank} has a pack length larger than capacity "
            f"{pack_capacity}."
        )
    _validate_boundaries(
        packing_meta.get("boundaries"),
        rank=rank,
        source_counts=source_counts,
        pack_lengths=pack_lengths,
    )
    return pack_count, pack_capacity


class NoOpPacker(Packer):
    """Validate and preserve metadata for physical packs built by Energon."""

    def packing_args(
        self, mb_tokens_key: str
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Disable NeMo-RL sequence and dynamic batch planning."""
        return None, None

    def pack(self, packing_input: PackingInput) -> PackingResult:
        """Validate one producer-placed metadata item per logical DP rank."""
        if not isinstance(packing_input, PlacedPackingInput):
            raise TypeError(
                "NoOpPacker accepts only PlacedPackingInput, got "
                f"{type(packing_input).__name__}."
            )
        if packing_input.dp_world <= 0:
            raise ValueError(
                f"NoOpPacker requires dp_world > 0, got {packing_input.dp_world}."
            )
        if len(packing_input.dp_metas) != packing_input.dp_world:
            raise ValueError(
                "Placed metadata must contain exactly one batch per DP rank: "
                f"got {len(packing_input.dp_metas)} batches for "
                f"dp_world={packing_input.dp_world}."
            )

        summaries = [
            _validate_meta(meta, rank=rank)
            for rank, meta in enumerate(packing_input.dp_metas)
        ]
        pack_counts = {pack_count for pack_count, _ in summaries}
        if len(pack_counts) != 1:
            raise ValueError(
                "Energon-packed DP ranks must have equal physical pack counts, got "
                f"{sorted(pack_counts)}."
            )
        capacities = {capacity for _, capacity in summaries}
        if len(capacities) != 1:
            raise ValueError(
                "Energon-packed DP ranks must use one pack capacity, got "
                f"{sorted(capacities)}."
            )
        return PackingResult(dp_metas=packing_input.dp_metas)


__all__ = [
    "ENERGON_PACKING_META_KEY",
    "ENERGON_PACKING_SCHEMA_VERSION",
    "NoOpPacker",
]
