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

"""Megatron-LM-style balanced greedy knapsack sequence packing.

Adapted from ``balanced_greedy_knapsack`` in
``examples/multimodal/data_loading/knapsacks.py`` in Megatron-LM at commit
6822175d92a40e0528be905aee50f5930cfa0c98. This is the algorithm the Nemotron
production launch script selects (``--packing-knapsack-algorithm
balanced_greedy_knapsack --packing-algorithm-parameters
balanced_knapsack_delta=5``).

Unlike ``greedy_knapsack``, which fills one bin to the brim before opening the
next, this sorts descending and always places into the least-full bin, so the
fill is spread evenly. It also pre-allocates ``ceil(total / capacity) + delta``
bins, which is why empty bins are a normal part of its output.

Two deliberate deviations from the reference, both forced by the SequencePacker
contract and noted here so the difference is not mistaken for a port error:

  * Oversized items. The reference prints a warning and silently drops any item
    larger than the capacity. ``SequencePacker.pack`` validates first and raises
    instead, so an oversized item never reaches this implementation. The skip
    branch is kept for parity when ``_pack_implementation`` is called directly.
  * Empty bins. The reference returns them; they are preserved here so bin
    counts match, and ``_adjust_bin_count`` may append more.
"""

from nemo_rl.data.packing.base import SequencePacker


class BalancedGreedyKnapsackPacker(SequencePacker):
    """Spread items across pre-allocated bins, always filling the emptiest."""

    def __init__(
        self,
        bin_capacity: int,
        collect_metrics: bool = False,
        min_bin_count: int | None = None,
        bin_count_multiple: int | None = None,
        balanced_knapsack_delta: int = 20,
    ) -> None:
        """Initialize the packer.

        Args:
            bin_capacity: Maximum cost of each bin.
            collect_metrics: Whether to collect metrics across packing calls.
            min_bin_count: Minimum number of non-empty bins.
            bin_count_multiple: Required multiple for the number of bins.
            balanced_knapsack_delta: Extra bins pre-allocated beyond the minimum
                needed to hold the total cost. The reference default is 20; the
                production launch script passes 5.

        Raises:
            ValueError: If ``balanced_knapsack_delta`` is negative.
        """
        super().__init__(
            bin_capacity=bin_capacity,
            collect_metrics=collect_metrics,
            min_bin_count=min_bin_count,
            bin_count_multiple=bin_count_multiple,
        )
        if balanced_knapsack_delta < 0:
            raise ValueError("balanced_knapsack_delta must be nonnegative")
        self.balanced_knapsack_delta = balanced_knapsack_delta

    def _pack_implementation(self, sequence_lengths: list[int]) -> list[list[int]]:
        # Descending by cost, keeping the source index so bins report positions.
        # A stable sort leaves equal costs in source order, matching the
        # reference's sort on cost alone.
        order = sorted(
            range(len(sequence_lengths)),
            key=lambda index: sequence_lengths[index],
            reverse=True,
        )

        total_length = sum(sequence_lengths)
        bin_count = (
            (total_length + self.bin_capacity - 1) // self.bin_capacity
            + self.balanced_knapsack_delta
        )
        bins: list[list[int]] = [[] for _ in range(bin_count)]
        bin_lengths: list[int] = [0] * bin_count

        bin_index = 0
        position = 0
        while position < len(order):
            source_index = order[position]
            length = sequence_lengths[source_index]
            if length > self.bin_capacity:
                # Unreachable through pack(), which validates first.
                position += 1
                continue
            if bin_lengths[bin_index] + length <= self.bin_capacity:
                bins[bin_index].append(source_index)
                bin_lengths[bin_index] += length
                position += 1
            else:
                # Nothing fits the emptiest bin, so open another.
                bins.append([])
                bin_lengths.append(0)
            # min() ties resolve to the lowest index, as in the reference.
            bin_index = bin_lengths.index(min(bin_lengths))

        return bins
