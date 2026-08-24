import pytest

from nemo_rl.data.packing import PackingAlgorithm, SequencePacker, get_packer


@pytest.mark.parametrize("algorithm", list(PackingAlgorithm))
def test_factory_builds_each_registered_algorithm(
    algorithm: PackingAlgorithm,
) -> None:
    assert isinstance(get_packer(algorithm, 8), SequencePacker)
    assert isinstance(get_packer(algorithm.value, 8), SequencePacker)


def test_factory_reports_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unknown packing algorithm"):
        get_packer("missing", 8)
