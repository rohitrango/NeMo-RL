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

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class EnergonSourceConfig(BaseModel, extra="allow"):
    """One prepared Energon dataset split."""

    path: str
    split: str
    virtual_epoch_length: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1)] | None = None


class EnergonLoaderConfig(BaseModel, extra="allow"):
    """Driver-side Energon settings for multimodal SFT."""

    num_workers: Annotated[int, Field(ge=0)] = 8
    shuffle_buffer_size: Annotated[int, Field(ge=0)] = 1000
    max_samples_per_sequence: None = None
    packing_buffer_size: None = None
    batch_grouping: Literal["auto"] = "auto"
    processor_adapter: Literal["hf_multimodal"] = "hf_multimodal"
    seed_offset: int = 0
    prefetch_factor: Annotated[int, Field(ge=1)] = 2
    checkpoint_every_sec: Annotated[float, Field(gt=0)] = 60.0
    watchdog_timeout_seconds: Annotated[float, Field(gt=0)] | None = 60.0
