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

"""Factory for shared sequence packing algorithms."""

import enum

from nemo_rl.data.packing.base import SequencePacker
from nemo_rl.data.packing.balanced_greedy_knapsack import (
    BalancedGreedyKnapsackPacker,
)
from nemo_rl.data.packing.concatenative import ConcatenativePacker
from nemo_rl.data.packing.first_fit_decreasing import FirstFitDecreasingPacker
from nemo_rl.data.packing.first_fit_shuffle import FirstFitShufflePacker
from nemo_rl.data.packing.greedy_knapsack import GreedyKnapsackPacker
from nemo_rl.data.packing.modified_first_fit_decreasing import (
    ModifiedFirstFitDecreasingPacker,
)


class PackingAlgorithm(enum.Enum):
    """Supported sequence packing algorithms."""

    CONCATENATIVE = "concatenative"
    FIRST_FIT_DECREASING = "first_fit_decreasing"
    FIRST_FIT_SHUFFLE = "first_fit_shuffle"
    MODIFIED_FIRST_FIT_DECREASING = "modified_first_fit_decreasing"
    GREEDY_KNAPSACK = "greedy_knapsack"
    BALANCED_GREEDY_KNAPSACK = "balanced_greedy_knapsack"


_PACKER_TYPES: dict[PackingAlgorithm, type[SequencePacker]] = {
    PackingAlgorithm.CONCATENATIVE: ConcatenativePacker,
    PackingAlgorithm.FIRST_FIT_DECREASING: FirstFitDecreasingPacker,
    PackingAlgorithm.FIRST_FIT_SHUFFLE: FirstFitShufflePacker,
    PackingAlgorithm.MODIFIED_FIRST_FIT_DECREASING: ModifiedFirstFitDecreasingPacker,
    PackingAlgorithm.GREEDY_KNAPSACK: GreedyKnapsackPacker,
    PackingAlgorithm.BALANCED_GREEDY_KNAPSACK: BalancedGreedyKnapsackPacker,
}


def get_packer(
    algorithm: PackingAlgorithm | str,
    bin_capacity: int,
    collect_metrics: bool = False,
    min_bin_count: int | None = None,
    bin_count_multiple: int | None = None,
    **packer_options: object,
) -> SequencePacker:
    """Build one packer from a stable enum value or algorithm key.

    ``packer_options`` carries algorithm-specific settings, such as
    ``balanced_knapsack_delta`` for ``balanced_greedy_knapsack``.
    """
    if isinstance(algorithm, str):
        try:
            algorithm = PackingAlgorithm(algorithm.lower())
        except ValueError as error:
            available = ", ".join(item.value for item in PackingAlgorithm)
            raise ValueError(
                f"Unknown packing algorithm: {algorithm}. Available algorithms: "
                f"{available}"
            ) from error
    packer_type = _PACKER_TYPES.get(algorithm)
    if packer_type is None:
        available = ", ".join(item.value for item in PackingAlgorithm)
        raise ValueError(
            f"Unknown packing algorithm: {algorithm}. Available algorithms: {available}"
        )
    return packer_type(
        bin_capacity,
        collect_metrics=collect_metrics,
        min_bin_count=min_bin_count,
        bin_count_multiple=bin_count_multiple,
        **packer_options,
    )


__all__ = ["PackingAlgorithm", "get_packer"]
