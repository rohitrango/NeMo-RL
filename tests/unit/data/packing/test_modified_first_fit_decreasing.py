from nemo_rl.data.packing import ModifiedFirstFitDecreasingPacker


def test_modified_first_fit_decreasing_preserves_equal_cost_source_order() -> None:
    bins = ModifiedFirstFitDecreasingPacker(10).pack([6, 6, 3, 3, 1, 1])

    assert bins == [[0, 2, 4], [1, 3, 5]]
