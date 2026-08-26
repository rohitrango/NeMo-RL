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

"""Run V1 SFT over the same rank-sharded Energon stream used by SFTv2."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from nemo_rl.data.energon import build_energon_sft_loader
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_DP_SIZE_ENV = "SFT_REFERENCE_DP_SIZE"


class _RankCombinedLoader:
    """Concatenate one local batch per logical rank in rank order."""

    def __init__(self, loaders: list[Any]) -> None:
        if not loaders:
            raise ValueError("At least one logical-rank loader is required.")
        lengths = [len(loader) for loader in loaders]
        if len(set(lengths)) != 1:
            raise ValueError(f"Logical-rank loader lengths differ: {lengths}.")
        self._loaders = loaders
        self._length = lengths[0]
        self._iteration_started = False

    def __iter__(self) -> Iterator[BatchedDataDict[Any]]:
        self._iteration_started = True
        iterators = [iter(loader) for loader in self._loaders]
        for step in range(self._length):
            batches = []
            for logical_rank, iterator in enumerate(iterators):
                try:
                    batches.append(next(iterator))
                except StopIteration as error:
                    raise RuntimeError(
                        f"Logical-rank loader {logical_rank} stopped before "
                        f"reference step {step}."
                    ) from error
            yield BatchedDataDict.from_batches(batches)

    def __len__(self) -> int:
        return self._length

    def state_dict(self) -> dict[str, Any]:
        return {
            "backend": "energon_rank_combined_reference",
            "format_version": 1,
            "logical_world_size": len(self._loaders),
            "loader_states": [loader.state_dict() for loader in self._loaders],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._iteration_started:
            raise RuntimeError("Loader state must be restored before iteration.")
        if state.get("backend") != "energon_rank_combined_reference":
            raise ValueError("Rank-combined reference loader state is invalid.")
        loader_states = state.get("loader_states")
        if not isinstance(loader_states, list) or len(loader_states) != len(
            self._loaders
        ):
            raise ValueError("Rank-combined reference loader state count differs.")
        for loader, loader_state in zip(
            self._loaders, loader_states, strict=True
        ):
            loader.load_state_dict(loader_state)


def _build_rank_combined_dataloaders(
    *,
    data_config: Mapping[str, Any],
    processor: Any,
    train_batch_size: int,
    val_batch_size: int,
    max_sequence_length: int,
) -> tuple[_RankCombinedLoader, None]:
    del val_batch_size
    logical_world_size = int(os.environ[_DP_SIZE_ENV])
    if logical_world_size <= 1:
        raise ValueError(f"{_DP_SIZE_ENV} must exceed one for the DP diagnostic.")
    if train_batch_size % logical_world_size != 0:
        raise ValueError(
            f"Global batch {train_batch_size} is not divisible by logical DP size "
            f"{logical_world_size}."
        )
    local_batch_size = train_batch_size // logical_world_size
    loaders = [
        build_energon_sft_loader(
            data_config=data_config,
            source=data_config["train"],
            processor=processor,
            batch_size=local_batch_size,
            max_sequence_length=max_sequence_length,
            split_role="train",
            logical_rank=logical_rank,
            logical_world_size=logical_world_size,
            placement_fingerprint=f"sft_v1_dp_reference_{logical_world_size}",
        )
        for logical_rank in range(logical_world_size)
    ]
    print(
        f"V1 reference data: {logical_world_size} Energon shards x "
        f"{local_batch_size} samples",
        flush=True,
    )
    return _RankCombinedLoader(loaders), None


def main() -> None:
    if _DP_SIZE_ENV not in os.environ:
        raise ValueError(f"Set {_DP_SIZE_ENV} to the V1 data-parallel size.")

    # SFT setup resolves this public loader builder at call time. Override it only
    # inside this diagnostic process, then run the ordinary V1 setup/train loop.
    import nemo_rl.data.energon as energon_api

    examples_dir = str(Path(__file__).resolve().parents[1] / "examples")
    sys.path.insert(0, examples_dir)
    try:
        from run_sft import main as run_sft
    finally:
        sys.path.remove(examples_dir)

    original_builder = energon_api.build_energon_sft_dataloaders
    energon_api.build_energon_sft_dataloaders = _build_rank_combined_dataloaders
    try:
        run_sft(is_vlm=True)
    finally:
        energon_api.build_energon_sft_dataloaders = original_builder


if __name__ == "__main__":
    main()
