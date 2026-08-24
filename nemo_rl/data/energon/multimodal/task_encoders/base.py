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

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from megatron.energon import Cooker, CrudeSample, DefaultTaskEncoder

from nemo_rl.data.energon.multimodal.packing import (
    EnergonPackingHooks,
    validate_packing_schema,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
    PackedSFTSample,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class BaseSFTTaskEncoder(
    DefaultTaskEncoder[
        CanonicalSFTSample,
        EncodedSFTSample,
        BatchedDataDict[Any],
        BatchedDataDict[Any],
    ],
    ABC,
):
    """Common SFT lifecycle and optional Energon packing forwarding."""

    sample_schema: ClassVar[str]

    def __init__(
        self,
        *,
        cooker_functions: Sequence[Callable[[CrudeSample], CanonicalSFTSample]],
        packing_hooks: EnergonPackingHooks[Any, Any, Any] | None,
    ) -> None:
        super().__init__()
        self.cookers = tuple(Cooker(cooker) for cooker in cooker_functions)
        self._packing_hooks: EnergonPackingHooks[Any, Any, Any] | None = None
        self._packing_is_bound = False
        self.register_packing(packing_hooks)

    def register_packing(
        self, hooks: EnergonPackingHooks[Any, Any, Any] | None
    ) -> None:
        """Bind the selected packing callbacks once during loader setup."""
        if self._packing_is_bound:
            raise RuntimeError("Energon packing hooks are already bound.")
        if hooks is not None:
            validate_packing_schema(self.sample_schema, hooks)
        self._packing_hooks = hooks
        self._packing_is_bound = True

    def _require_packing(self) -> EnergonPackingHooks[Any, Any, Any]:
        if self._packing_hooks is None:
            raise RuntimeError("No Energon packing implementation is configured.")
        return self._packing_hooks

    def select_samples_to_pack(self, samples: list[Any]) -> list[list[Any]]:
        """Forward pack selection to the configured packing implementation."""
        return self._require_packing().select_samples_to_pack(samples)

    def pack_selected_samples(self, samples: list[Any]) -> Any:
        """Forward physical pack construction to the configured implementation."""
        return self._require_packing().pack_selected_samples(samples)

    @abstractmethod
    def preencode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Encode one sample before optional pack selection."""

    @abstractmethod
    def postencode_sample(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        """Finish the selected sample before physical packing."""

    @abstractmethod
    def batch(
        self, samples: list[EncodedSFTSample | PackedSFTSample]
    ) -> BatchedDataDict[Any]:
        """Combine encoded samples into one minibatch."""

    @abstractmethod
    def encode_batch(self, batch: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        """Finish one minibatch before the loader emits it."""


__all__ = ["BaseSFTTaskEncoder"]
