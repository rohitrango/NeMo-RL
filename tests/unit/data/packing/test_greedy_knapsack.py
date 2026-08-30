import pytest

from nemo_rl.data.packing import GreedyKnapsackPacker


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
