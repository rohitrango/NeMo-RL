import random

from nemo_rl.data.packing import FirstFitShufflePacker


def test_first_fit_shuffle_uses_seeded_source_order() -> None:
    random.seed(11)
    first = FirstFitShufflePacker(10).pack([5, 5, 5, 5])
    random.seed(11)
    second = FirstFitShufflePacker(10).pack([5, 5, 5, 5])

    assert first == second
    assert sorted(index for bin_indexes in first for index in bin_indexes) == [
        0,
        1,
        2,
        3,
    ]
