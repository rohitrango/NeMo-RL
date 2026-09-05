from nemo_rl.data.packing import ConcatenativePacker


def test_concatenative_preserves_source_order() -> None:
    assert ConcatenativePacker(8).pack([4, 4, 2, 6]) == [[0, 1], [2, 3]]
