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

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

PreT = TypeVar("PreT")
PostT = TypeVar("PostT")
PackedT = TypeVar("PackedT")

ENERGON_PACKED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EnergonPackingHooks(Generic[PreT, PostT, PackedT]):
    """One immutable pair of Energon packing callbacks."""

    key: str
    version: str
    sample_schema: str
    select_samples_to_pack: Callable[[list[PreT]], list[list[PreT]]]
    pack_selected_samples: Callable[[list[PostT]], PackedT]


def validate_packing_schema(
    task_encoder_schema: str, hooks: EnergonPackingHooks[Any, Any, Any]
) -> None:
    """Reject task-encoder and packing implementations with different schemas."""
    if task_encoder_schema != hooks.sample_schema:
        raise ValueError(
            "Task encoder and Energon packing sample schemas differ: "
            f"{task_encoder_schema!r} != {hooks.sample_schema!r}."
        )


__all__ = [
    "ENERGON_PACKED_SCHEMA_VERSION",
    "EnergonPackingHooks",
    "validate_packing_schema",
]
