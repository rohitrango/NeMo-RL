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
"""Existing NeMo-RL metadata packing behavior behind the Packer boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Protocol

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.models.policy.packing.base import (
    GlobalPackingInput,
    Packer,
    PackingInput,
    PackingResult,
    PlacedPackingInput,
)


class ShardMetaFn(Protocol):
    """Callable shape used for metadata-only DP assignment."""

    def __call__(
        self,
        meta: KVBatchMeta,
        *,
        dp_world: int,
        batch_size: Optional[int] = None,
        sequence_packing_args: Optional[dict[str, Any]] = None,
        dynamic_batching_args: Optional[dict[str, Any]] = None,
    ) -> tuple[list[KVBatchMeta], Optional[list[int]]]: ...


class NeMoRLPacker(Packer):
    """Apply NeMo-RL fixed, sequence, or dynamic metadata assignment."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        use_dynamic_batches: bool,
        dynamic_batching_args: Optional[Mapping[str, Any]],
        use_sequence_packing: bool,
        sequence_packing_args: Optional[Mapping[str, Any]],
        shard_meta: ShardMetaFn = shard_meta_for_dp,
    ) -> None:
        if use_dynamic_batches and use_sequence_packing:
            raise ValueError(
                "NeMoRLPacker accepts at most one of dynamic batching and "
                "sequence packing."
            )
        if use_dynamic_batches and dynamic_batching_args is None:
            raise ValueError(
                "dynamic_batching_args are required when dynamic batching is enabled."
            )
        if use_sequence_packing and sequence_packing_args is None:
            raise ValueError(
                "sequence_packing_args are required when sequence packing is enabled."
            )
        self._cfg = cfg
        self._use_dynamic_batches = use_dynamic_batches
        self._dynamic_batching_args = dynamic_batching_args
        self._use_sequence_packing = use_sequence_packing
        self._sequence_packing_args = sequence_packing_args
        self._shard_meta = shard_meta

    def packing_args(
        self, mb_tokens_key: str
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Resolve packing arguments for the requested policy stage."""
        if self._use_dynamic_batches:
            args = dict(self._dynamic_batching_args or {})
            args["max_tokens_per_microbatch"] = self._cfg["dynamic_batching"][
                mb_tokens_key
            ]
            return None, args
        if self._use_sequence_packing:
            args = dict(self._sequence_packing_args or {})
            args["max_tokens_per_microbatch"] = self._cfg["sequence_packing"][
                mb_tokens_key
            ]
            return args, None
        return None, None

    def pack(self, packing_input: PackingInput) -> PackingResult:
        """Plan metadata without moving tensor payloads."""
        if packing_input.dp_world <= 0:
            raise ValueError(
                f"NeMoRLPacker requires dp_world > 0, got {packing_input.dp_world}."
            )
        sequence_args, dynamic_args = self.packing_args(packing_input.mb_tokens_key)
        if isinstance(packing_input, GlobalPackingInput):
            dp_metas, unsorted_indices = self._shard_meta(
                packing_input.meta,
                dp_world=packing_input.dp_world,
                batch_size=packing_input.batch_size,
                sequence_packing_args=sequence_args,
                dynamic_batching_args=dynamic_args,
            )
            return PackingResult(dp_metas=dp_metas, unsorted_indices=unsorted_indices)
        if isinstance(packing_input, PlacedPackingInput):
            if len(packing_input.dp_metas) != packing_input.dp_world:
                raise ValueError(
                    "Placed metadata must contain exactly one batch per DP rank: "
                    f"got {len(packing_input.dp_metas)} batches for "
                    f"dp_world={packing_input.dp_world}."
                )
            if sequence_args is not None or dynamic_args is not None:
                raise ValueError(
                    "Stage 1 placed metadata supports fixed batches only. Disable "
                    "NeMo-RL sequence packing and dynamic batching."
                )
            return PackingResult(dp_metas=list(packing_input.dp_metas))
        raise TypeError(
            f"NeMoRLPacker does not support input type {type(packing_input).__name__}."
        )
