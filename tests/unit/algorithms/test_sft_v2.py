from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.sft_v2 import (
    SFTSingleControllerActor,
    SFTV2SaveState,
    _max_train_steps,
)
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data_plane import KVBatchMeta

_ACTOR_CLS = SFTSingleControllerActor.__ray_metadata__.modified_class


def _envelope(rank: int, *, source_count: int = 1) -> StepEnvelope:
    return StepEnvelope(
        meta=KVBatchMeta(
            partition_id=f"p{rank}",
            task_name="train",
            sample_ids=[f"s{rank}"],
            fields=["input_ids"],
            sequence_lengths=[8],
        ),
        logical_rank=rank,
        logical_world_size=2,
        source_ids=tuple(
            f"source-{rank}-{source_index}" for source_index in range(source_count)
        ),
        field_names=("input_ids",),
        sequence_lengths=(8,),
        load_seconds=0.1 + rank * 0.1,
        valid_tokens=4,
    )


def _controller() -> object:
    controller = object.__new__(_ACTOR_CLS)
    controller._trainer = MagicMock()
    controller._trainer.finish_train_step.return_value = {
        "loss": 1.0,
        "grad_norm": 0.5,
        "all_mb_metrics": {},
    }
    controller._master_config = SimpleNamespace()
    controller._save_state = SFTV2SaveState(0, 0, 0, "hash")
    controller._load_envelopes = MagicMock(return_value=[_envelope(0), _envelope(1)])
    controller._owner_call = MagicMock(return_value=[None, None])
    controller._loss_fn = object()
    return controller


def test_train_step_orders_split_policy_lifecycle_and_commit() -> None:
    controller = _controller()
    controller._load_envelopes.return_value = [
        _envelope(0, source_count=2),
        _envelope(1),
    ]

    metrics = controller._run_train_step()

    controller._trainer.begin_train_step.assert_called_once_with(controller._loss_fn)
    controller._trainer.train_placed_microbatches.assert_called_once()
    controller._trainer.finish_train_step.assert_called_once_with()
    controller._owner_call.assert_called_once_with("commit_sft_batch")
    assert controller._save_state.total_steps == 1
    assert controller._save_state.consumed_samples == 3
    assert metrics["valid_tokens"] == 8
    assert metrics["source_samples"] == 3
    assert metrics["physical_packs"] == 2


def test_train_step_aborts_policy_and_loader_on_training_failure() -> None:
    controller = _controller()
    controller._trainer.train_placed_microbatches.side_effect = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        controller._run_train_step()

    controller._trainer.abort_train_step.assert_called_once_with()
    controller._owner_call.assert_any_call("abort_sft_batch")
    assert controller._save_state.total_steps == 0


def test_restore_rejects_changed_placement() -> None:
    from nemo_rl.algorithms.sft_v2 import _restore_save_state

    with pytest.raises(ValueError, match="placement changed"):
        _restore_save_state(
            {"placement_hash": "old", "total_steps": 3}, placement_hash="new"
        )


def test_max_steps_is_bounded_by_virtual_epochs() -> None:
    config = SimpleNamespace(
        sft=SimpleNamespace(max_num_steps=100, max_num_epochs=3),
        data={
            "train": {
                "path": "/dataset",
                "split": "train",
                "virtual_epoch_length": 7,
            }
        },
    )

    assert _max_train_steps(config) == 21
