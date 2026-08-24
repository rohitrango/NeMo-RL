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
"""Policy packing implementations."""

from collections.abc import Mapping
from typing import Any, Optional

from nemo_rl.models.policy.packing.base import (
    GlobalPackingInput,
    Packer,
    PackingInput,
    PackingResult,
    PlacedPackingInput,
)
from nemo_rl.models.policy.packing.nemo_rl import NeMoRLPacker, ShardMetaFn
from nemo_rl.models.policy.packing.no_op import (
    ENERGON_PACKING_META_KEY,
    ENERGON_PACKING_SCHEMA_VERSION,
    NoOpPacker,
)


def resolve_packer(
    name: str,
    *,
    cfg: Mapping[str, Any],
    use_dynamic_batches: bool,
    dynamic_batching_args: Optional[Mapping[str, Any]],
    use_sequence_packing: bool,
    sequence_packing_args: Optional[Mapping[str, Any]],
    shard_meta: Optional[ShardMetaFn] = None,
) -> Packer:
    """Resolve a policy packer by name."""
    if name == "no_op":
        return NoOpPacker()
    if name != "nemo_rl":
        raise ValueError(
            f"Unknown policy packer {name!r}. Supported packers are "
            "'nemo_rl' and 'no_op'."
        )
    if shard_meta is None:
        return NeMoRLPacker(
            cfg=cfg,
            use_dynamic_batches=use_dynamic_batches,
            dynamic_batching_args=dynamic_batching_args,
            use_sequence_packing=use_sequence_packing,
            sequence_packing_args=sequence_packing_args,
        )
    return NeMoRLPacker(
        cfg=cfg,
        use_dynamic_batches=use_dynamic_batches,
        dynamic_batching_args=dynamic_batching_args,
        use_sequence_packing=use_sequence_packing,
        sequence_packing_args=sequence_packing_args,
        shard_meta=shard_meta,
    )


__all__ = [
    "ENERGON_PACKING_META_KEY",
    "ENERGON_PACKING_SCHEMA_VERSION",
    "GlobalPackingInput",
    "NeMoRLPacker",
    "NoOpPacker",
    "Packer",
    "PackingInput",
    "PackingResult",
    "PlacedPackingInput",
    "resolve_packer",
]
