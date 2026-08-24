from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.models.policy.packing import (
    ENERGON_PACKING_META_KEY,
    ENERGON_PACKING_SCHEMA_VERSION,
    GlobalPackingInput,
    NoOpPacker,
    PlacedPackingInput,
)


def _meta(
    *,
    rank: int,
    pack_lengths: list[int] | None = None,
    source_counts: list[int] | None = None,
    pack_capacity: int = 16,
) -> KVBatchMeta:
    if pack_lengths is None:
        pack_lengths = [12, 8]
    if source_counts is None:
        source_counts = [2, 1]
    boundaries = [
        {
            "cu_seqlens": [0, 5, 11],
            "cu_seqlens_padded": [0, 6, 12],
        },
        {
            "cu_seqlens": [0, 7],
            "cu_seqlens_padded": [0, 8],
        },
    ][: len(pack_lengths)]
    return KVBatchMeta(
        partition_id="train",
        task_name=f"rank-{rank}",
        sample_ids=[f"rank-{rank}-pack-{index}" for index in range(len(pack_lengths))],
        fields=["input_ids", "pixel_values", "cu_seqlens_padded"],
        sequence_lengths=list(pack_lengths),
        extra_info={
            "producer": f"loader-{rank}",
            ENERGON_PACKING_META_KEY: {
                "schema_version": ENERGON_PACKING_SCHEMA_VERSION,
                "pack_count": len(pack_lengths),
                "source_count": sum(source_counts),
                "source_counts": list(source_counts),
                "pack_lengths": list(pack_lengths),
                "pack_capacity": pack_capacity,
                "boundaries": boundaries,
            },
        },
    )


def _placed(metas: list[KVBatchMeta]) -> PlacedPackingInput:
    return PlacedPackingInput(
        dp_metas=metas,
        dp_world=len(metas),
        mb_tokens_key="train_mb_tokens",
    )


def _packing_meta(meta: KVBatchMeta) -> dict[str, Any]:
    value = meta.extra_info[ENERGON_PACKING_META_KEY]
    assert isinstance(value, dict)
    return value


def test_preserves_placed_metadata_and_returns_no_packing_args() -> None:
    metas = [_meta(rank=0), _meta(rank=1)]
    fields = [meta.fields for meta in metas]
    before = deepcopy(metas)
    packer = NoOpPacker()

    result = packer.pack(_placed(metas))

    assert result.dp_metas is metas
    assert all(
        result_meta is input_meta
        for result_meta, input_meta in zip(result.dp_metas, metas)
    )
    assert result.dp_metas == before
    assert [meta.fields for meta in result.dp_metas] == fields
    assert result.unsorted_indices is None
    assert packer.packing_args("train_mb_tokens") == (None, None)


def test_accepts_different_source_counts_with_equal_pack_schedules() -> None:
    first = _meta(rank=0)
    second = _meta(rank=1, source_counts=[1, 1])
    _packing_meta(second)["boundaries"] = [
        {"cu_seqlens": [0, 11], "cu_seqlens_padded": [0, 12]},
        {"cu_seqlens": [0, 7], "cu_seqlens_padded": [0, 8]},
    ]

    assert NoOpPacker().pack(_placed([first, second])).dp_metas == [first, second]


def test_rejects_global_input() -> None:
    meta = _meta(rank=0)

    with pytest.raises(TypeError, match="only PlacedPackingInput"):
        NoOpPacker().pack(
            GlobalPackingInput(
                meta=meta,
                dp_world=1,
                batch_size=2,
                mb_tokens_key="train_mb_tokens",
            )
        )


def test_requires_one_meta_per_dp_rank() -> None:
    with pytest.raises(ValueError, match="exactly one batch per DP rank"):
        NoOpPacker().pack(
            PlacedPackingInput(
                dp_metas=[_meta(rank=0)],
                dp_world=2,
                mb_tokens_key="train_mb_tokens",
            )
        )


def test_rejects_missing_packed_schema() -> None:
    meta = _meta(rank=0)
    del meta.extra_info[ENERGON_PACKING_META_KEY]

    with pytest.raises(ValueError, match="missing 'energon_packing' metadata"):
        NoOpPacker().pack(_placed([meta]))


def test_rejects_unknown_schema_version() -> None:
    meta = _meta(rank=0)
    _packing_meta(meta)["schema_version"] = 2

    with pytest.raises(ValueError, match="unsupported Energon packing schema"):
        NoOpPacker().pack(_placed([meta]))


def test_rejects_unequal_physical_pack_counts() -> None:
    short = _meta(rank=1, pack_lengths=[12], source_counts=[2])

    with pytest.raises(ValueError, match="equal physical pack counts"):
        NoOpPacker().pack(_placed([_meta(rank=0), short]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_count", 4, "source_counts sum"),
        ("source_counts", [3, 1], "source_counts sum"),
        ("pack_lengths", [11, 8], "must match KVBatchMeta.sequence_lengths"),
        ("pack_capacity", 10, "pack length larger than capacity"),
        ("boundaries", None, "one item per physical pack"),
    ],
)
def test_rejects_inconsistent_packing_summary(
    field: str, value: Any, message: str
) -> None:
    meta = _meta(rank=0)
    _packing_meta(meta)[field] = value

    with pytest.raises(ValueError, match=message):
        NoOpPacker().pack(_placed([meta]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cu_seqlens", [1, 5, 11], "must start at zero"),
        ("cu_seqlens", [0, 5, 5], "strictly increasing"),
        ("cu_seqlens_padded", [0, 6, 11], "must end at pack length"),
        ("cu_seqlens_padded", [0, 4, 12], "source length larger"),
    ],
)
def test_rejects_invalid_pack_boundaries(
    field: str, value: list[int], message: str
) -> None:
    meta = _meta(rank=0)
    boundaries = _packing_meta(meta)["boundaries"]
    assert isinstance(boundaries, list)
    boundaries[0][field] = value

    with pytest.raises(ValueError, match=message):
        NoOpPacker().pack(_placed([meta]))


def test_rejects_different_pack_capacities() -> None:
    with pytest.raises(ValueError, match="use one pack capacity"):
        NoOpPacker().pack(
            _placed([_meta(rank=0, pack_capacity=16), _meta(rank=1, pack_capacity=32)])
        )
