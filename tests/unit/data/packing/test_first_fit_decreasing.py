from nemo_rl.data.packing import FirstFitDecreasingPacker


def test_first_fit_decreasing_preserves_equal_cost_source_order() -> None:
    assert FirstFitDecreasingPacker(10).pack([5, 5, 5, 5]) == [[0, 1], [2, 3]]


def test_first_fit_decreasing_uses_first_available_bin() -> None:
    assert FirstFitDecreasingPacker(8).pack([6, 5, 4, 3]) == [
        [0],
        [1, 3],
        [2],
    ]
