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
import os
import traceback
import urllib.parse
from collections.abc import Callable
from typing import Any, Iterator, Literal, Mapping, Protocol, cast

import torch
from megatron.energon import (
    Cooker,
    CrudeSample,
    FileStoreCachePool,
    SourceInfo,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
)
from megatron.energon.epathlib import epath as energon_epath

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonSourceConfig
from nemo_rl.data.energon.multimodal.packing import build_packing_hooks
from nemo_rl.data.energon.multimodal.registry import (
    COOKER_REGISTRY,
    PACKING_REGISTRY,
    TASK_ENCODER_REGISTRY,
    selected_registry_identity,
)
from nemo_rl.data.energon.multimodal.task_encoders import (
    BaseSFTTaskEncoder,
    build_processor_adapter,
)
from nemo_rl.data.energon.multimodal.types import CanonicalSFTSample
from nemo_rl.data.packing import SequencePacker
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_V1_STATE_FORMAT_VERSION = 1
_V2_STATE_FORMAT_VERSION = 2
_FIRST_SAMPLE_ASSERTION = False


def _set_nvdataset_cache_dir(path: str) -> None:
    """Point Energon's ``dss://`` resolution at ``path`` for this process.

    Energon snapshots ``NVDATASET_CACHE_DIR`` into a module global at import
    time (``megatron/energon/epathlib/epath.py``), and every ``dss://`` lookup
    reads that global rather than the environment. This module imports
    ``megatron.energon`` at the top, so setting the env var alone would be a
    silent no-op -- rebind the global too. The env var is still set so forked
    Energon loader workers inherit it.
    """
    os.environ["NVDATASET_CACHE_DIR"] = path
    energon_epath.NVDATASET_CACHE_DIR = energon_epath.EPath(path)


def compact_sample_error_handler(
    exception: Exception,
    sample: Any | list[Any],
    sources: list[SourceInfo] | None = None,
) -> None:
    """Log a sample-processing error and let Energon request another sample."""
    global _FIRST_SAMPLE_ASSERTION

    if isinstance(exception, AssertionError):
        if sources is None:
            print(f"Assertion error in sample {str(sample)[:100]}: {exception}")
            return

        data = [
            {
                "dataset_path": str(source.dataset_path),
                "index": source.index,
                "shard_name": source.shard_name,
                "file_names": list(source.file_names),
            }
            for source in sources
        ]
        url = (
            "vscode://nvidia.energon-sample-viewer/open?data="
            f"{urllib.parse.quote(json.dumps(data))}"
        )
        print(f"Assertion error: {exception}")
        print(f"(Ctrl+)Click to view sample in energon viewer: {url}")
        if _FIRST_SAMPLE_ASSERTION:
            print(
                "If not installed yet, install energon sample viewer from "
                "https://gitlab-master.nvidia.com/lvoegtle/"
                "vscode-energon-sample-viewer"
            )
            _FIRST_SAMPLE_ASSERTION = False
        return

    print("Ignoring error processing sample:")
    traceback.print_exc()


class SFTDataLoader(Protocol):
    """Iterator and state methods consumed by the SFT algorithm."""

    def __iter__(self) -> Iterator[BatchedDataDict[Any]]: ...

    def __len__(self) -> int: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class EnergonSFTDataLoader:
    """Expose Energon rank state through NeMo-RL's dataloader interface."""

    def __init__(
        self, loader: Any, *, fingerprint: str, state_format_version: int = 1
    ) -> None:
        self._loader = loader
        self._fingerprint = fingerprint
        self._state_format_version = state_format_version
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
            "format_version": self._state_format_version,
            "fingerprint": self._fingerprint,
            # The outer NeMo-RL checkpoint stays compatible with torch.load's
            # weights-only default. Energon classes are decoded after validation.
            "loader_state": torch.frombuffer(
                bytearray(buffer.getvalue()), dtype=torch.uint8
            ).clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._iteration_started:
            raise RuntimeError(
                "Energon loader state must be restored before iteration."
            )
        if state.get("backend") != "energon":
            raise ValueError("Cannot restore non-Energon state into an Energon loader.")
        if state.get("format_version") != self._state_format_version:
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
        if (
            not isinstance(loader_state, torch.Tensor)
            or loader_state.dtype != torch.uint8
        ):
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
        config = EnergonLoaderConfig.model_validate(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Energon loader configuration.") from error
    if config.topology_mapper != "default":
        raise ValueError(
            f"Unknown data-loader topology mapper {config.topology_mapper!r}."
        )
    configured_encoder_options = config.task_encoder.options.model_fields_set
    if config.task_encoder.name == "generic_sft":
        if configured_encoder_options:
            raise ValueError(
                f"Task encoder {config.task_encoder.name!r} has no configurable "
                "options."
            )
    if not config.cookers:
        raise ValueError("At least one Energon cooker must be configured.")
    if any(cooker.options for cooker in config.cookers):
        raise ValueError("The Stage 1 generic conversation cooker has no options.")
    fallback_cookers = [
        index
        for index, cooker in enumerate(config.cookers)
        if cooker.has_subflavors is None
    ]
    if len(fallback_cookers) > 1:
        raise ValueError(
            "At most one Energon cooker can omit has_subflavors when several "
            "cookers are configured."
        )
    if fallback_cookers and fallback_cookers[0] != len(config.cookers) - 1:
        raise ValueError("An Energon fallback cooker must be the final cooker.")
    filters = [
        cooker.has_subflavors
        for cooker in config.cookers
        if cooker.has_subflavors is not None
    ]
    if any(current in filters[:index] for index, current in enumerate(filters)):
        raise ValueError("Energon cooker has_subflavors filters must be unique.")
    TASK_ENCODER_REGISTRY.resolve_for_model_family(
        config.task_encoder.name,
        model_family=config.model_family,
    )
    for cooker in config.cookers:
        COOKER_REGISTRY.resolve_for_model_family(
            cooker.name,
            model_family=config.model_family,
        )
    return config


def _v1_loader_projection(config: EnergonLoaderConfig) -> dict[str, Any]:
    """Preserve legacy V1 state while identifying Stage 3 components."""
    projection: dict[str, Any] = {
        "num_workers": config.num_workers,
        "shuffle_buffer_size": config.shuffle_buffer_size,
        "max_samples_per_sequence": config.max_samples_per_sequence,
        "packing_buffer_size": config.packing_buffer_size,
        "batch_grouping": config.batch_grouping,
        "processor_adapter": config.processor_adapter,
        "seed_offset": config.seed_offset,
        "prefetch_factor": config.prefetch_factor,
        "checkpoint_every_sec": config.checkpoint_every_sec,
        "watchdog_timeout_seconds": config.watchdog_timeout_seconds,
    }
    legacy_components = (
        config.task_encoder.name == "generic_sft"
        and not config.task_encoder.options.model_fields_set
        and config.task_encoder.packing is None
        and len(config.cookers) == 1
        and config.cookers[0].name == "generic_conversation"
        and not config.cookers[0].options
        and config.cookers[0].has_subflavors is None
    )
    if legacy_components:
        return projection

    projection["stage3"] = {
        "model_family": config.model_family,
        "task_encoder": config.task_encoder.model_dump(mode="json"),
        "cookers": [cooker.model_dump(mode="json") for cooker in config.cookers],
        "registries": selected_registry_identity(
            task_encoder=config.task_encoder.name,
            cookers=[cooker.name for cooker in config.cookers],
            packing=(
                None
                if config.task_encoder.packing is None
                else config.task_encoder.packing.name
            ),
        ),
    }
    return projection


def _v1_fingerprint(
    *,
    source: EnergonSourceConfig,
    loader_config: EnergonLoaderConfig,
    adapter_fingerprint: str,
    split_role: str,
) -> str:
    payload = {
        "source": source.model_dump(mode="json"),
        "loader": _v1_loader_projection(loader_config),
        "adapter": adapter_fingerprint,
        "split_role": split_role,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _v2_fingerprint(
    *,
    source: EnergonSourceConfig,
    loader_config: EnergonLoaderConfig,
    adapter_fingerprint: str,
    split_role: str,
    logical_rank: int,
    logical_world_size: int,
    placement_fingerprint: str,
) -> str:
    packing = loader_config.task_encoder.packing
    payload = {
        "state_format_version": _V2_STATE_FORMAT_VERSION,
        "source": source.model_dump(mode="json"),
        "loader": loader_config.model_dump(mode="json"),
        "adapter": adapter_fingerprint,
        "split_role": split_role,
        "topology": {
            "mapper": loader_config.topology_mapper,
            "placement": placement_fingerprint,
            "logical_rank": logical_rank,
            "logical_world_size": logical_world_size,
        },
        "registries": selected_registry_identity(
            task_encoder=loader_config.task_encoder.name,
            cookers=[cooker.name for cooker in loader_config.cookers],
            packing=None if packing is None else packing.name,
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _worker_config(
    config: EnergonLoaderConfig, *, logical_rank: int, logical_world_size: int
) -> WorkerConfig:
    if logical_world_size <= 0:
        raise ValueError("Logical data world size must be positive.")
    if not 0 <= logical_rank < logical_world_size:
        raise ValueError(
            f"Logical data rank {logical_rank} is outside world size "
            f"{logical_world_size}."
        )
    return WorkerConfig(
        rank=logical_rank,
        world_size=logical_world_size,
        num_workers=config.num_workers,
        seed_offset=config.seed_offset,
        global_error_handler=compact_sample_error_handler,
        restore_error_handler=compact_sample_error_handler,
    )


def _task_encoder(
    *,
    loader_config: EnergonLoaderConfig,
    adapter: Any,
    include_source_ids: bool,
    tokenizer: Any | None = None,
    only_unmask_final: bool = False,
) -> BaseSFTTaskEncoder:
    cooker_functions = [
        Cooker(
            cast(
                Callable[[CrudeSample], CanonicalSFTSample],
                COOKER_REGISTRY.resolve(cooker.name),
            ),
            has_subflavors=cooker.has_subflavors,
        )
        for cooker in loader_config.cookers
    ]
    encoder_type = cast(
        Any, TASK_ENCODER_REGISTRY.resolve(loader_config.task_encoder.name)
    )
    encoder_options: dict[str, Any] = {}
    if loader_config.task_encoder.name == "nemotron_multimodal":
        encoder_options = loader_config.task_encoder.options.model_dump()
    packing_config = loader_config.task_encoder.packing
    packing_hooks = None
    if packing_config is not None:
        packer_type = cast(
            type[SequencePacker],
            PACKING_REGISTRY.resolve(packing_config.name),
        )
        packing_hooks = build_packing_hooks(
            packing_config.options,
            algorithm=packing_config.name,
            version=PACKING_REGISTRY.identity(packing_config.name)["version"],
            packer_type=packer_type,
        )
    return cast(
        BaseSFTTaskEncoder,
        encoder_type(
            adapter=adapter,
            cooker_functions=cooker_functions,
            packing_hooks=packing_hooks,
            include_source_ids=include_source_ids,
            tokenizer=tokenizer,
            only_unmask_final=only_unmask_final,
            **encoder_options,
        ),
    )


def _build_energon_sft_loader(
    *,
    data_config: Mapping[str, Any],
    source: EnergonSourceConfig,
    processor: Any,
    batch_size: int,
    max_sequence_length: int,
    split_role: Literal["train", "validation"],
    logical_rank: int,
    logical_world_size: int,
    placement_fingerprint: str | None,
    state_format_version: Literal[1, 2],
    only_unmask_final: bool,
) -> EnergonSFTDataLoader:
    if processor is None:
        raise ValueError("data.backend=energon requires a multimodal processor.")
    if batch_size <= 0:
        raise ValueError("Energon SFT batch size must be positive.")
    if state_format_version == _V2_STATE_FORMAT_VERSION and not placement_fingerprint:
        raise ValueError("SFTv2 requires a non-empty placement fingerprint.")

    loader_config = _loader_config(data_config["energon"])
    if loader_config.nvdataset_cache_dir is not None:
        _set_nvdataset_cache_dir(loader_config.nvdataset_cache_dir)
    if (
        state_format_version == _V1_STATE_FORMAT_VERSION
        and loader_config.task_encoder.packing is not None
    ):
        raise ValueError("Energon-owned packing requires the SFTv2 loader path.")
    if (
        loader_config.task_encoder.packing is not None
        and loader_config.task_encoder.packing.options.max_sequence_length
        != max_sequence_length
    ):
        raise ValueError(
            "Energon pack capacity must match the SFT maximum sequence length."
        )
    adapter = build_processor_adapter(
        processor_adapter=loader_config.processor_adapter,
        processor=processor,
        max_sequence_length=max_sequence_length,
        add_bos=data_config.get("add_bos", True),
        add_eos=data_config.get("add_eos", True),
        add_generation_prompt=data_config.get("add_generation_prompt", False),
    )
    task_encoder = _task_encoder(
        loader_config=loader_config,
        adapter=adapter,
        include_source_ids=state_format_version == _V2_STATE_FORMAT_VERSION,
        tokenizer=processor.tokenizer,
        only_unmask_final=only_unmask_final,
    )
    worker_config = _worker_config(
        loader_config,
        logical_rank=logical_rank,
        logical_world_size=logical_world_size,
    )

    if split_role == "train":
        if source.virtual_epoch_length <= 0:
            raise ValueError(
                "Energon training requires train.virtual_epoch_length in batches."
            )
        dataset = get_train_dataset(
            source.path,
            split_part=source.split,
            worker_config=worker_config,
            batch_size=batch_size,
            batch_drop_last=True,
            packing_buffer_size=(
                None
                if loader_config.task_encoder.packing is None
                else loader_config.task_encoder.packing.buffer_size
            ),
            shuffle_buffer_size=(
                loader_config.shuffle_buffer_size if data_config["shuffle"] else None
            ),
            max_samples_per_sequence=None,
            virtual_epoch_length=source.virtual_epoch_length,
            task_encoder=task_encoder,
        )
    else:
        dataset = get_val_dataset(
            source.path,
            split_part=source.split,
            worker_config=worker_config,
            batch_size=batch_size,
            batch_drop_last=False,
            packing_buffer_size=(
                None
                if loader_config.task_encoder.packing is None
                else loader_config.task_encoder.packing.buffer_size
            ),
            limit=source.limit,
            task_encoder=task_encoder,
        )

    # FileStoreCachePool defaults to max_cache_size_gbytes=1024 (1 TB) and
    # num_workers=8. On a node whose whole memory cgroup is 900Gi that default
    # is effectively unbounded, and the cache-worker IPC shows up as Shmem
    # charged to the cgroup. The reference bounds it explicitly
    # (energon-megatron-lm examples/multimodal/iter_data.py:
    #  FileStoreCachePool(num_workers=8, max_cache_size_gbytes=8, method="raw")).
    # Let the recipe set both, and bound the default instead of inheriting 1 TB.
    def _loader_opt(name, default=None):
        if isinstance(loader_config, dict):
            return loader_config.get(name, default)
        return getattr(loader_config, name, default)

    _cache_kwargs = {"method": "raw"}
    _cache_gb = _loader_opt("cache_pool_max_gbytes", None)
    if _cache_gb is not None:
        _cache_kwargs["max_cache_size_gbytes"] = _cache_gb
    _cache_workers = _loader_opt("cache_pool_num_workers", 1)
    if _cache_workers is not None:
        _cache_kwargs["num_workers"] = _cache_workers

    cache_pool = (
        FileStoreCachePool(**_cache_kwargs)
        if any(cooker.need_cache for cooker in task_encoder.cookers)
        else None
    )
    # Reference (energon-megatron-lm examples/multimodal/iter_data.py) leaves
    # this at its high default (100000) rather than the SavableDataLoader
    # default -- gc.collect() on the shuffle/packing buffers is expensive at
    # buffer_size=5000+, so only force it rarely. Let the recipe override.
    _gc_every_n_steps = _loader_opt("gc_collect_every_n_steps", 100000)

    loader = get_savable_loader(
        dataset,
        cache_pool=cache_pool,
        checkpoint_every_sec=loader_config.checkpoint_every_sec,
        prefetch_factor=loader_config.prefetch_factor,
        watchdog_timeout_seconds=loader_config.watchdog_timeout_seconds,
        gc_collect_every_n_steps=_gc_every_n_steps,
        fail_on_timeout=True,
    )
    if state_format_version == _V1_STATE_FORMAT_VERSION:
        fingerprint = _v1_fingerprint(
            source=source,
            loader_config=loader_config,
            adapter_fingerprint=adapter.fingerprint,
            split_role=split_role,
        )
    else:
        assert placement_fingerprint is not None
        fingerprint = _v2_fingerprint(
            source=source,
            loader_config=loader_config,
            adapter_fingerprint=adapter.fingerprint,
            split_role=split_role,
            logical_rank=logical_rank,
            logical_world_size=logical_world_size,
            placement_fingerprint=placement_fingerprint,
        )
    return EnergonSFTDataLoader(
        loader,
        fingerprint=fingerprint,
        state_format_version=state_format_version,
    )


def build_energon_sft_loader(
    *,
    data_config: Mapping[str, Any],
    source: Mapping[str, Any] | EnergonSourceConfig,
    processor: Any,
    batch_size: int,
    max_sequence_length: int,
    split_role: Literal["train", "validation"],
    logical_rank: int,
    logical_world_size: int,
    placement_fingerprint: str,
    only_unmask_final: bool = False,
) -> EnergonSFTDataLoader:
    """Build one V2 loader for an explicit logical data shard and split."""
    if "energon" not in data_config:
        raise ValueError("data.backend=energon requires a data.energon block.")
    resolved_source = _source_config(source, name=split_role)
    return _build_energon_sft_loader(
        data_config=data_config,
        source=resolved_source,
        processor=processor,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        split_role=split_role,
        logical_rank=logical_rank,
        logical_world_size=logical_world_size,
        placement_fingerprint=placement_fingerprint,
        state_format_version=_V2_STATE_FORMAT_VERSION,
        only_unmask_final=only_unmask_final,
    )


def build_energon_sft_dataloaders(
    *,
    data_config: Mapping[str, Any],
    processor: Any,
    train_batch_size: int,
    val_batch_size: int,
    max_sequence_length: int,
) -> tuple[EnergonSFTDataLoader, EnergonSFTDataLoader | None]:
    """Build V1 driver-owned loaders through the shared rank-aware path."""
    if "energon" not in data_config:
        raise ValueError("data.backend=energon requires a data.energon block.")
    if isinstance(data_config.get("train"), list):
        raise ValueError(
            "Energon v1 accepts one train path; use an Energon metadataset to "
            "blend sources."
        )

    train_source = _source_config(data_config.get("train"), name="train")
    train_loader = _build_energon_sft_loader(
        data_config=data_config,
        source=train_source,
        processor=processor,
        batch_size=train_batch_size,
        max_sequence_length=max_sequence_length,
        split_role="train",
        logical_rank=0,
        logical_world_size=1,
        placement_fingerprint=None,
        state_format_version=_V1_STATE_FORMAT_VERSION,
        only_unmask_final=False,
    )

    validation = data_config.get("validation")
    if validation is None:
        return train_loader, None
    if isinstance(validation, list):
        raise ValueError(
            "Energon v1 accepts one validation path; use an Energon metadataset "
            "to combine sources."
        )
    val_source = _source_config(validation, name="validation")
    val_loader = _build_energon_sft_loader(
        data_config=data_config,
        source=val_source,
        processor=processor,
        batch_size=val_batch_size,
        max_sequence_length=max_sequence_length,
        split_role="validation",
        logical_rank=0,
        logical_world_size=1,
        placement_fingerprint=None,
        state_format_version=_V1_STATE_FORMAT_VERSION,
        only_unmask_final=False,
    )
    return train_loader, val_loader


__all__ = [
    "EnergonSFTDataLoader",
    "SFTDataLoader",
    "build_energon_sft_loader",
    "build_energon_sft_dataloaders",
]
