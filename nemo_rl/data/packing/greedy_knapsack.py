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

"""Megatron-LM-style greedy knapsack sequence packing.

Adapted from ``examples/multimodal/data_loading/knapsacks.py`` in Megatron-LM
at commit 6822175d92a40e0528be905aee50f5930cfa0c98.
"""

from bisect import bisect_right

from nemo_rl.data.packing.base import SequencePacker


class GreedyKnapsackPacker(SequencePacker):
    """Repeatedly select the largest remaining item that fits the active bin."""

    def _pack_implementation(self, sequence_lengths: list[int]) -> list[list[int]]:
        # A descending source index in the ascending list makes removal from the
        # right preserve original source order when multiple items have equal cost.
        remaining = sorted(
            (
                (length, -source_index, source_index)
                for source_index, length in enumerate(sequence_lengths)
            )
        )
        bins: list[list[int]] = []
        while remaining:
            current_bin: list[int] = []
            remaining_capacity = self.bin_capacity
            while True:
                fit_index = bisect_right(
                    remaining,
                    (remaining_capacity, 1, len(sequence_lengths)),
                )
                if fit_index == 0:
                    break
                length, _, source_index = remaining.pop(fit_index - 1)
                remaining_capacity -= length
                current_bin.append(source_index)
            bins.append(current_bin)
        return bins


__all__ = ["GreedyKnapsackPacker"]
