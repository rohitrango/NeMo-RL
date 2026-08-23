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
from typing import Any

from megatron.energon import Sample, edataclass


@dataclass(frozen=True)
class MediaRef:
    """One ordered media occurrence in a conversation."""

    modality: str
    value: Any


@edataclass
class CanonicalSFTSample(Sample):
    """Model-neutral conversation produced by an Energon cooker."""

    messages: list[dict[str, Any]]
    media: list[MediaRef]
    tools: list[dict[str, Any]] | None


@edataclass
class EncodedSFTSample(Sample):
    """One tokenized message log before batching."""

    message_log: list[dict[str, Any]]
    length: int
    loss_multiplier: float
    group_key: tuple[Any, ...]
    sample_key: str


__all__ = ["CanonicalSFTSample", "EncodedSFTSample", "MediaRef"]
