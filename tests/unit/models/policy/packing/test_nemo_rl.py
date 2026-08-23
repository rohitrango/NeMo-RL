from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.models.policy.packing import (
    GlobalPackingInput,
    NeMoRLPacker,
    PlacedPackingInput,
)


def _meta(n: int = 8) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=[f"sample-{index}" for index in range(n)],
        fields=["input_ids", "token_mask"],
        sequence_lengths=[8, 63, 17, 41, 29, 55, 12, 34][:n],
        extra_info={"source": "test"},
    )


def _config() -> dict[str, Any]:
    return {
        "sequence_packing": {
            "logprob_mb_tokens": 128,
            "train_mb_tokens": 256,
        },
        "dynamic_batching": {
            "logprob_mb_tokens": 128,
            "train_mb_tokens": 256,
        },
    }


def _packer(mode: str, shard_meta: Any = shard_meta_for_dp) -> NeMoRLPacker:
    return NeMoRLPacker(
        cfg=_config(),
        use_dynamic_batches=mode == "dynamic",
        dynamic_batching_args=(
            {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": 8,
                "max_tokens_per_microbatch": 0,
            }
            if mode == "dynamic"
            else None
        ),
        use_sequence_packing=mode == "sequence",
        sequence_packing_args=(
            {
                "algorithm": "modified_first_fit_decreasing",
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": 8,
            }
            if mode == "sequence"
            else None
        ),
        shard_meta=shard_meta,
    )


def _meta_signature(meta: KVBatchMeta) -> tuple[Any, ...]:
    return (
        meta.partition_id,
        meta.task_name,
        meta.sample_ids,
        meta.fields,
        meta.sequence_lengths,
        meta.extra_info,
    )


@pytest.mark.parametrize("mode", ["fixed", "sequence", "dynamic"])
def test_global_assignment_matches_existing_shard_meta(mode: str) -> None:
    meta = _meta()
    packer = _packer(mode)
    sequence_args, dynamic_args = packer.packing_args("train_mb_tokens")

    expected_metas, expected_unsorted = shard_meta_for_dp(
        meta,
        dp_world=2,
        batch_size=8,
        sequence_packing_args=sequence_args,
        dynamic_batching_args=dynamic_args,
    )
    result = packer.pack(
        GlobalPackingInput(
            meta=meta,
            dp_world=2,
            batch_size=8,
            mb_tokens_key="train_mb_tokens",
        )
    )

    assert [_meta_signature(value) for value in result.dp_metas] == [
        _meta_signature(value) for value in expected_metas
    ]
    assert result.unsorted_indices == expected_unsorted


def test_stage_token_limit_is_added_without_mutating_base_args() -> None:
    packer = _packer("dynamic")

    _, logprob_args = packer.packing_args("logprob_mb_tokens")
    _, train_args = packer.packing_args("train_mb_tokens")

    assert logprob_args is not None
    assert train_args is not None
    assert logprob_args["max_tokens_per_microbatch"] == 128
    assert train_args["max_tokens_per_microbatch"] == 256


def test_placed_fixed_batches_keep_rank_order_and_fields() -> None:
    shard_meta = MagicMock()
    packer = _packer("fixed", shard_meta=shard_meta)
    dp_metas = [_meta(2), _meta(2)]
    dp_metas[0].fields = ["input_ids", "pixel_values"]
    dp_metas[1].fields = ["input_ids", "image_grid_thw"]

    result = packer.pack(
        PlacedPackingInput(
            dp_metas=dp_metas,
            dp_world=2,
            mb_tokens_key="train_mb_tokens",
        )
    )

    assert result.dp_metas == dp_metas
    assert [meta.fields for meta in result.dp_metas] == [
        ["input_ids", "pixel_values"],
        ["input_ids", "image_grid_thw"],
    ]
    assert result.unsorted_indices is None
    shard_meta.assert_not_called()


def test_placed_batches_require_one_batch_per_dp_rank() -> None:
    with pytest.raises(ValueError, match="one batch per DP rank"):
        _packer("fixed").pack(
            PlacedPackingInput(
                dp_metas=[_meta(2)],
                dp_world=2,
                mb_tokens_key="train_mb_tokens",
            )
        )


@pytest.mark.parametrize("mode", ["sequence", "dynamic"])
def test_stage1_placed_batches_reject_global_packing_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="fixed batches only"):
        _packer(mode).pack(
            PlacedPackingInput(
                dp_metas=[_meta(2), _meta(2)],
                dp_world=2,
                mb_tokens_key="train_mb_tokens",
            )
        )
