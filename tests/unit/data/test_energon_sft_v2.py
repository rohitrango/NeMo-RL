from __future__ import annotations

from unittest.mock import MagicMock, patch

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonSourceConfig
from nemo_rl.data.energon.sft_dataloader import (
    _v2_fingerprint,
    _worker_config,
    build_energon_sft_dataloaders,
)
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env


def test_worker_config_uses_logical_data_rank() -> None:
    worker = _worker_config(
        EnergonLoaderConfig(num_workers=3),
        logical_rank=2,
        logical_world_size=4,
    )

    assert worker.rank == 2
    assert worker.world_size == 4
    assert worker.num_workers == 3


def test_v2_fingerprint_identifies_each_logical_shard() -> None:
    source = EnergonSourceConfig(path="/dataset", split="train", virtual_epoch_length=8)
    loader = EnergonLoaderConfig()
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


def test_v1_builder_uses_shared_loader_as_rank_zero_of_one() -> None:
    train_loader = MagicMock()
    data_config = {
        "energon": {},
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
