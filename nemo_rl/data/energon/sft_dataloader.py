# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import io
import json
from typing import Any, Iterator, Mapping, Protocol

from megatron.energon import (
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    reraise_exception,
)
import torch

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonSourceConfig
from nemo_rl.data.energon.sft import EnergonSFTTaskEncoder, build_processor_adapter
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_STATE_FORMAT_VERSION = 1


class SFTDataLoader(Protocol):
    """Iterator and state methods consumed by the SFT algorithm."""

    def __iter__(self) -> Iterator[BatchedDataDict[Any]]: ...

    def __len__(self) -> int: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class EnergonSFTDataLoader:
    """Expose Energon rank state through NeMo-RL's dataloader interface."""

    def __init__(self, loader: Any, *, fingerprint: str) -> None:
        self._loader = loader
        self._fingerprint = fingerprint
        self._iteration_started = False

    def __iter__(self) -> Iterator[BatchedDataDict[Any]]:
        self._iteration_started = True
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def state_dict(self) -> dict[str, Any]:
        buffer = io.BytesIO()
        torch.save(self._loader.save_state_rank(), buffer)
        return {
            "backend": "energon",
            "format_version": _STATE_FORMAT_VERSION,
            "fingerprint": self._fingerprint,
            # The outer NeMo-RL checkpoint stays compatible with torch.load's
            # weights-only default. Energon state classes are decoded only
            # after the backend and fingerprint checks below.
            "loader_state": torch.frombuffer(
                bytearray(buffer.getvalue()), dtype=torch.uint8
            ).clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._iteration_started:
            raise RuntimeError("Energon loader state must be restored before iteration.")
        if state.get("backend") != "energon":
            raise ValueError("Cannot restore non-Energon state into an Energon loader.")
        if state.get("format_version") != _STATE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported Energon loader state format "
                f"{state.get('format_version')!r}."
            )
        if state.get("fingerprint") != self._fingerprint:
            raise ValueError(
                "Energon loader fingerprint mismatch; dataset, processor, or loader "
                "settings changed."
            )
        loader_state = state.get("loader_state")
        if not isinstance(loader_state, torch.Tensor) or loader_state.dtype != torch.uint8:
            raise ValueError("Energon loader state payload is invalid.")
        decoded = torch.load(
            io.BytesIO(loader_state.cpu().numpy().tobytes()),
            weights_only=False,
        )
        self._loader.restore_state_rank(decoded)


def _source_config(value: Any, *, name: str) -> EnergonSourceConfig:
    try:
        return EnergonSourceConfig.model_validate(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Energon {name} source configuration.") from error


def _loader_config(value: Any) -> EnergonLoaderConfig:
    try:
        return EnergonLoaderConfig.model_validate(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Energon loader configuration.") from error


def _fingerprint(
    *,
    source: EnergonSourceConfig,
    loader_config: EnergonLoaderConfig,
    adapter_fingerprint: str,
    split_role: str,
) -> str:
    payload = {
        "source": source.model_dump(mode="json"),
        "loader": loader_config.model_dump(mode="json"),
        "adapter": adapter_fingerprint,
        "split_role": split_role,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _task_encoder(
    *,
    adapter: Any,
) -> EnergonSFTTaskEncoder:
    return EnergonSFTTaskEncoder(
        adapter=adapter,
    )


def _worker_config(config: EnergonLoaderConfig) -> WorkerConfig:
    return WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=config.num_workers,
        seed_offset=config.seed_offset,
        global_error_handler=reraise_exception,
    )


def build_energon_sft_dataloaders(
    *,
    data_config: Mapping[str, Any],
    processor: Any,
    train_batch_size: int,
    val_batch_size: int,
    max_sequence_length: int,
) -> tuple[EnergonSFTDataLoader, EnergonSFTDataLoader | None]:
    """Build driver-owned train and validation loaders from prepared datasets."""
    if processor is None:
        raise ValueError("data.backend=energon requires a multimodal processor.")
    if "energon" not in data_config:
        raise ValueError("data.backend=energon requires a data.energon block.")
    if isinstance(data_config.get("train"), list):
        raise ValueError(
            "Energon v1 accepts one train path; use an Energon metadataset to blend sources."
        )

    loader_config = _loader_config(data_config["energon"])
    train_source = _source_config(data_config.get("train"), name="train")
    if train_source.virtual_epoch_length <= 0:
        raise ValueError(
            "Energon training requires train.virtual_epoch_length in batches."
        )
    adapter = build_processor_adapter(
        processor_adapter=loader_config.processor_adapter,
        processor=processor,
        max_sequence_length=max_sequence_length,
        add_bos=data_config.get("add_bos", True),
        add_eos=data_config.get("add_eos", True),
        add_generation_prompt=data_config.get("add_generation_prompt", False),
    )

    train_dataset = get_train_dataset(
        train_source.path,
        split_part=train_source.split,
        worker_config=_worker_config(loader_config),
        batch_size=train_batch_size,
        batch_drop_last=True,
        packing_buffer_size=None,
        shuffle_buffer_size=(
            loader_config.shuffle_buffer_size if data_config["shuffle"] else None
        ),
        max_samples_per_sequence=None,
        virtual_epoch_length=train_source.virtual_epoch_length,
        task_encoder=_task_encoder(
            adapter=adapter,
        ),
    )
    train_loader = get_savable_loader(
        train_dataset,
        checkpoint_every_sec=loader_config.checkpoint_every_sec,
        prefetch_factor=loader_config.prefetch_factor,
        watchdog_timeout_seconds=loader_config.watchdog_timeout_seconds,
        fail_on_timeout=True,
    )
    wrapped_train = EnergonSFTDataLoader(
        train_loader,
        fingerprint=_fingerprint(
            source=train_source,
            loader_config=loader_config,
            adapter_fingerprint=adapter.fingerprint,
            split_role="train",
        ),
    )

    validation = data_config.get("validation")
    if validation is None:
        return wrapped_train, None
    if isinstance(validation, list):
        raise ValueError(
            "Energon v1 accepts one validation path; use an Energon metadataset "
            "to combine sources."
        )
    val_source = _source_config(validation, name="validation")
    val_dataset = get_val_dataset(
        val_source.path,
        split_part=val_source.split,
        worker_config=_worker_config(loader_config),
        batch_size=val_batch_size,
        batch_drop_last=False,
        packing_buffer_size=None,
        limit=val_source.limit,
        task_encoder=_task_encoder(
            adapter=adapter,
        ),
    )
    val_loader = get_savable_loader(
        val_dataset,
        checkpoint_every_sec=loader_config.checkpoint_every_sec,
        prefetch_factor=loader_config.prefetch_factor,
        watchdog_timeout_seconds=loader_config.watchdog_timeout_seconds,
        fail_on_timeout=True,
    )
    return wrapped_train, EnergonSFTDataLoader(
        val_loader,
        fingerprint=_fingerprint(
            source=val_source,
            loader_config=loader_config,
            adapter_fingerprint=adapter.fingerprint,
            split_role="validation",
        ),
    )


__all__ = [
    "EnergonSFTDataLoader",
    "SFTDataLoader",
    "build_energon_sft_dataloaders",
]
