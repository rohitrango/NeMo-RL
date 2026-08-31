from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

# sft_dataloader imports megatron.energon and sft_worker imports megatron.core,
# both of which ship only in the `mcore` extra. importorskip must run before the
# imports below: the mcore mark is applied in pytest_collection_modifyitems,
# too late to prevent a collection error.
pytest.importorskip("megatron.energon")
pytest.importorskip("megatron.core")

pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.config import (  # noqa: E402
    EnergonLoaderConfig,
    EnergonSourceConfig,
)
from nemo_rl.data.energon.sft_dataloader import (  # noqa: E402
    _v2_fingerprint,
    _worker_config,
    build_energon_sft_dataloaders,
)
from nemo_rl.data_plane import KVBatchMeta  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.distributed.ray_actor_environment_registry import (  # noqa: E402
    get_actor_python_env,
)


def test_worker_config_uses_logical_data_rank() -> None:
    worker = _worker_config(
        EnergonLoaderConfig(model_family="qwen", num_workers=3),
        logical_rank=2,
        logical_world_size=4,
    )

    assert worker.rank == 2
    assert worker.world_size == 4
    assert worker.num_workers == 3


def test_v2_fingerprint_identifies_each_logical_shard() -> None:
    source = EnergonSourceConfig(path="/dataset", split="train", virtual_epoch_length=8)
    loader = EnergonLoaderConfig(model_family="qwen")
    common = {
        "source": source,
        "loader_config": loader,
        "adapter_fingerprint": "processor",
        "split_role": "train",
        "logical_world_size": 2,
        "placement_fingerprint": "placement",
    }

    rank_zero = _v2_fingerprint(logical_rank=0, **common)
    rank_one = _v2_fingerprint(logical_rank=1, **common)

    assert rank_zero != rank_one
    assert rank_zero == _v2_fingerprint(logical_rank=0, **common)

    nemotron = loader.model_copy(update={"model_family": "nemotron"})
    assert rank_zero != _v2_fingerprint(
        logical_rank=0,
        **{**common, "loader_config": nemotron},
    )


def test_v1_builder_uses_shared_loader_as_rank_zero_of_one() -> None:
    train_loader = MagicMock()
    data_config = {
        "energon": {"model_family": "qwen"},
        "shuffle": True,
        "train": {
            "path": "/dataset",
            "split": "train",
            "virtual_epoch_length": 8,
        },
        "validation": None,
    }
    with patch(
        "nemo_rl.data.energon.sft_dataloader._build_energon_sft_loader",
        return_value=train_loader,
    ) as build:
        actual, validation = build_energon_sft_dataloaders(
            data_config=data_config,
            processor=object(),
            train_batch_size=8,
            val_batch_size=4,
            max_sequence_length=512,
        )

    assert actual is train_loader
    assert validation is None
    assert build.call_args.kwargs["logical_rank"] == 0
    assert build.call_args.kwargs["logical_world_size"] == 1
    assert build.call_args.kwargs["state_format_version"] == 1


def test_sft_v2_worker_uses_megatron_worker_environment() -> None:
    assert get_actor_python_env(
        "nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker"
    ) == get_actor_python_env(
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    )


def test_sft_v2_worker_publishes_sequence_alignment() -> None:
    from nemo_rl.data.energon.sft_worker import SFTMegatronPolicyWorker

    worker_cls = SFTMegatronPolicyWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._sft_loader = MagicMock()
    worker._sft_loader_iterator = iter([{"message_log": []}])
    worker._sft_active_envelope = None
    worker._sft_logical_rank = 0
    worker._sft_logical_world_size = 2
    worker._sft_next_batch_index = 0
    worker.tokenizer = MagicMock(pad_token_id=0)
    worker._dp_client = MagicMock()

    prepared = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "input_lengths": torch.tensor([3, 2], dtype=torch.int32),
            "token_mask": torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "sample_mask": torch.ones(2),
        }
    )
    worker._dp_client.put_samples.return_value = KVBatchMeta(
        partition_id="sft_v2_dp0_batch0",
        task_name=None,
        sample_ids=["sft_v2_dp0_batch0_row0", "sft_v2_dp0_batch0_row1"],
        fields=list(prepared.keys()),
        sequence_lengths=[3, 2],
        extra_info={"generation": 7},
    )

    with patch(
        "nemo_rl.data.energon.sft_worker.prepare_sft_batch",
        return_value=prepared,
    ):
        envelope = worker.load_next_sft_batch(
            only_unmask_final=False,
            make_sequence_length_divisible_by=4,
        )

    assert envelope.meta.extra_info == {
        "generation": 7,
        "pad_to_multiple": 4,
    }
