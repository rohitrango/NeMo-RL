from __future__ import annotations

from unittest.mock import MagicMock

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.models.policy.packing import PackingResult, PlacedPackingInput
from nemo_rl.models.policy.tq_policy import TQPolicy


def _meta(rank: int, fields: list[str]) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="loader",
        sample_ids=[f"rank-{rank}-sample-0", f"rank-{rank}-sample-1"],
        fields=fields,
        sequence_lengths=[8 + rank, 16 + 8 * rank],
        extra_info={"logical_dp_rank": rank},
    )


def _policy() -> tuple[TQPolicy, MagicMock, MagicMock]:
    policy = object.__new__(TQPolicy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy._router_replay_enabled = False
    policy.flops_tracker = None
    policy.sharding_annotations = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = 2
    worker_group = MagicMock()
    policy.worker_group = worker_group
    packer = MagicMock()
    packer.packing_args.return_value = (None, None)
    packer.pack.side_effect = lambda value: PackingResult(dp_metas=list(value.dp_metas))
    policy.packer = packer
    return policy, worker_group, packer


def test_train_placed_microbatches_keeps_fields_and_replica_delivery() -> None:
    policy, worker_group, packer = _policy()
    dp_metas = [
        _meta(0, ["input_ids", "pixel_values"]),
        _meta(1, ["input_ids", "image_grid_thw"]),
    ]

    assert policy.train_placed_microbatches(dp_metas) is None

    packing_input = packer.pack.call_args.args[0]
    assert isinstance(packing_input, PlacedPackingInput)
    assert packing_input.dp_world == 2
    assert [meta.fields for meta in packing_input.dp_metas] == [
        ["input_ids", "pixel_values"],
        ["input_ids", "image_grid_thw"],
    ]
    assert [meta.task_name for meta in packing_input.dp_metas] == ["train", "train"]
    assert [
        meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] for meta in packing_input.dp_metas
    ] == [24, 24]

    dispatch = worker_group.run_all_workers_sharded_data.call_args
    assert dispatch.args[0] == "train_microbatch_presharded"
    assert dispatch.kwargs["meta"] == packing_input.dp_metas
    assert dispatch.kwargs["in_sharded_axes"] == ["data_parallel"]
    assert dispatch.kwargs["replicate_on_axes"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert dispatch.kwargs["output_is_replicated"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    worker_group.get_all_worker_results.assert_called_once()
