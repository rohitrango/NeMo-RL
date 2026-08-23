from __future__ import annotations

import pytest

from nemo_rl.models.policy.packing import NeMoRLPacker, Packer, resolve_packer


def _resolve(name: str) -> Packer:
    return resolve_packer(
        name,
        cfg={},
        use_dynamic_batches=False,
        dynamic_batching_args=None,
        use_sequence_packing=False,
        sequence_packing_args=None,
    )


def test_resolver_returns_nemo_rl_packer() -> None:
    assert isinstance(_resolve("nemo_rl"), NeMoRLPacker)


def test_resolver_rejects_unknown_packer() -> None:
    with pytest.raises(ValueError, match="Stage 1 supports only 'nemo_rl'"):
        _resolve("energon")
