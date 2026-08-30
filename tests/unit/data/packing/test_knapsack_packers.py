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

"""Tests for the Megatron-LM-style knapsack packers."""

import pytest

from nemo_rl.data.packing import BalancedGreedyKnapsackPacker, GreedyKnapsackPacker


def test_greedy_knapsack_selects_largest_remaining_item_that_fits() -> None:
    assert GreedyKnapsackPacker(10).pack([6, 5, 4, 3, 2]) == [
        [0, 2],
        [1, 3, 4],
    ]


def test_greedy_knapsack_preserves_equal_cost_source_order() -> None:
    assert GreedyKnapsackPacker(10).pack([5, 5, 5, 5]) == [[0, 1], [2, 3]]


def test_greedy_knapsack_rejects_oversized_item() -> None:
    with pytest.raises(ValueError, match="exceeds bin capacity"):
        GreedyKnapsackPacker(10).pack([11])


def test_balanced_greedy_knapsack_spreads_fill_across_bins() -> None:
    # greedy_knapsack fills one bin to the brim and returns [[0, 2], [1, 3, 4]];
    # this spreads the same items evenly instead.
    packer = BalancedGreedyKnapsackPacker(10, balanced_knapsack_delta=0)

    assert packer.pack([6, 5, 4, 3, 2]) == [[0, 3], [1, 2], [4]]


def test_balanced_greedy_knapsack_preallocates_delta_empty_bins() -> None:
    packer = BalancedGreedyKnapsackPacker(10, balanced_knapsack_delta=2)

    assert packer.pack([10]) == [[0], [], []]


def test_balanced_greedy_knapsack_rejects_oversized_item() -> None:
    with pytest.raises(ValueError, match="exceeds bin capacity"):
        BalancedGreedyKnapsackPacker(10).pack([11])


def test_balanced_greedy_knapsack_rejects_negative_delta() -> None:
    with pytest.raises(ValueError, match="balanced_knapsack_delta must be nonnegative"):
        BalancedGreedyKnapsackPacker(10, balanced_knapsack_delta=-1)
