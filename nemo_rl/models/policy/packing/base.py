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
"""Policy-side metadata packing interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from nemo_rl.data_plane import KVBatchMeta


@dataclass(frozen=True)
class PackingInput:
    """Common inputs required for policy metadata packing."""

    dp_world: int
    mb_tokens_key: str


@dataclass(frozen=True)
class GlobalPackingInput(PackingInput):
    """One global metadata batch that must be assigned across DP ranks."""

    meta: KVBatchMeta
    batch_size: Optional[int]


@dataclass(frozen=True)
class PlacedPackingInput(PackingInput):
    """Metadata batches that producers have already assigned to DP ranks."""

    dp_metas: list[KVBatchMeta]


@dataclass(frozen=True)
class PackingResult:
    """Per-DP metadata and an optional inverse assignment permutation."""

    dp_metas: list[KVBatchMeta]
    unsorted_indices: Optional[list[int]] = None


class Packer(ABC):
    """Plan policy metadata assignments for global or placed inputs."""

    @abstractmethod
    def packing_args(
        self, mb_tokens_key: str
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Return sequence-packing and dynamic-batching arguments."""

    @abstractmethod
    def pack(self, packing_input: PackingInput) -> PackingResult:
        """Return one metadata batch per logical DP rank."""
