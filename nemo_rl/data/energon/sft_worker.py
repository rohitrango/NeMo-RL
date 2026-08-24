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

"""Colocated Energon loader extension for Megatron policy workers."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from typing import Any, Mapping, Optional

import ray
import torch
from megatron.core import parallel_state

from nemo_rl.algorithms.sft import prepare_sft_batch
from nemo_rl.data.energon.multimodal.packing import ENERGON_PACKED_SCHEMA_VERSION
from nemo_rl.data.energon.multimodal.packing.prepare import (
    prepare_energon_packed_batch,
)
from nemo_rl.data.energon.sft_dataloader import (
    EnergonSFTDataLoader,
    build_energon_sft_loader,
)
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data_plane.adapters.local import local_batch_to_tensordict
from nemo_rl.data_plane.schema import MICRO_BATCH_INDICES, MICRO_BATCH_LENGTHS
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.models.policy.packing import ENERGON_PACKING_META_KEY
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class SFTMegatronPolicyWorker(MegatronPolicyWorkerImpl):
    """Megatron policy worker with an Energon loader on each DP owner."""

    def __init__(self, *args: Any, processor: Any = None, **kwargs: Any) -> None:
        self._sft_processor = processor
        self._sft_loader: Optional[EnergonSFTDataLoader] = None
        self._sft_loader_iterator: Any = None
        self._sft_active_envelope: Optional[StepEnvelope] = None
        self._sft_next_batch_index = 0
        self._sft_logical_rank: Optional[int] = None
        self._sft_logical_world_size: Optional[int] = None
        super().__init__(*args, **kwargs)

    def setup_sft_dataloader(
        self,
        *,
        data_config: Mapping[str, Any],
        batch_size: int,
        max_sequence_length: int,
        placement_fingerprint: str,
        restored_state: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Build the train loader on the TP0/PP0/CP0 rank of this DP replica."""
        if not self._is_replica_leader():
            return False
        if self._sft_loader is not None:
            raise RuntimeError("The SFT Energon loader is already configured.")
        if self._sft_processor is None:
            raise ValueError("SFTv2 requires a multimodal processor on policy workers.")

        logical_rank = parallel_state.get_data_parallel_rank()
        logical_world_size = parallel_state.get_data_parallel_world_size()
        self._sft_loader = build_energon_sft_loader(
            data_config=data_config,
            source=data_config["train"],
            processor=self._sft_processor,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            split_role="train",
            logical_rank=logical_rank,
            logical_world_size=logical_world_size,
            placement_fingerprint=placement_fingerprint,
        )
        if restored_state is not None:
            self._sft_loader.load_state_dict(restored_state)
        self._sft_loader_iterator = iter(self._sft_loader)
        self._sft_logical_rank = logical_rank
        self._sft_logical_world_size = logical_world_size
        return True

    def load_next_sft_batch(
        self,
        *,
        only_unmask_final: bool,
        make_sequence_length_divisible_by: int,
    ) -> StepEnvelope:
        """Load, prepare, and publish one batch into this process's local store."""
        if self._sft_loader is None or self._sft_loader_iterator is None:
            raise RuntimeError("The SFT Energon loader is not configured on this rank.")
        if self._sft_active_envelope is not None:
            raise RuntimeError(
                "Commit or abort the active SFT batch before loading again."
            )
        if self._sft_logical_rank is None or self._sft_logical_world_size is None:
            raise RuntimeError("The SFT logical loader identity is missing.")

        started = time.monotonic()
        batch = next(self._sft_loader_iterator)
        packed_schema_version = batch.get("packed_schema_version")
        energon_packed = packed_schema_version is not None
        if (
            energon_packed
            and packed_schema_version != ENERGON_PACKED_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported Energon packed SFT schema version "
                f"{packed_schema_version!r}; expected "
                f"{ENERGON_PACKED_SCHEMA_VERSION}."
            )
        if energon_packed:
            prepared = prepare_energon_packed_batch(
                batch,
                tokenizer=self.tokenizer,
                only_unmask_final=only_unmask_final,
            )
        else:
            prepared = prepare_sft_batch(
                batch,
                tokenizer=self.tokenizer,
                only_unmask_final=only_unmask_final,
                make_sequence_length_divisible_by=(
                    make_sequence_length_divisible_by
                ),
            )
        load_seconds = time.monotonic() - started
        batch_size = prepared.size
        source_ids = self._source_ids(prepared, batch_size=batch_size)
        partition_id = (
            f"sft_v2_dp{self._sft_logical_rank}_batch{self._sft_next_batch_index}"
        )
        sample_ids = [f"{partition_id}_row{row}" for row in range(batch_size)]
        fields = local_batch_to_tensordict(prepared, batch_size=batch_size)
        field_names = list(fields.keys())
        client = self._require_dp_client()
        client.register_partition(
            partition_id=partition_id,
            fields=field_names,
            num_samples=batch_size,
            consumer_tasks=["train"],
        )
        tags = self._source_tags(prepared, batch_size=batch_size)
        published_meta = client.put_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            fields=fields,
            tags=tags,
        )

        lengths_tensor = prepared["input_lengths"]
        lengths = tuple(int(value) for value in lengths_tensor.tolist())
        sample_mask = prepared["sample_mask"]
        valid_tokens = int(
            (sample_mask.unsqueeze(-1) * prepared["token_mask"][:, 1:]).sum().item()
        )
        extra_info = dict(published_meta.extra_info)
        if make_sequence_length_divisible_by > 1:
            extra_info["pad_to_multiple"] = int(make_sequence_length_divisible_by)
        if energon_packed:
            extra_info[ENERGON_PACKING_META_KEY] = self._packing_metadata(prepared)
            # The local fetch path trusts these producer-supplied boundaries
            # and skips its NeMo-RL bin planner. The Megatron prepacked path
            # then consumes one physical pack per microbatch.
            extra_info[MICRO_BATCH_INDICES] = [
                [[index, index + 1] for index in range(batch_size)]
            ]
            extra_info[MICRO_BATCH_LENGTHS] = [list(lengths)]
        envelope = StepEnvelope(
            meta=replace(
                published_meta,
                task_name="train",
                extra_info=extra_info,
            ),
            logical_rank=self._sft_logical_rank,
            logical_world_size=self._sft_logical_world_size,
            source_ids=source_ids,
            field_names=tuple(field_names),
            sequence_lengths=lengths,
            field_fingerprints=self._field_fingerprints(prepared),
            load_seconds=load_seconds,
            valid_tokens=valid_tokens,
        )
        self._sft_active_envelope = envelope
        self._sft_next_batch_index += 1
        return envelope

    def commit_sft_batch(self) -> None:
        """Release the active process-local batch after a successful step."""
        envelope = self._require_active_envelope()
        self._require_dp_client().clear_samples(
            sample_ids=envelope.meta.sample_ids,
            partition_id=envelope.meta.partition_id,
        )
        self._sft_active_envelope = None

    def abort_sft_batch(self) -> None:
        """Release the active batch after a failed policy step."""
        if self._sft_active_envelope is None:
            return
        self.commit_sft_batch()

    def sft_dataloader_state_dict(self) -> dict[str, Any]:
        """Capture this logical loader state after its batch is committed."""
        if self._sft_loader is None:
            raise RuntimeError("The SFT Energon loader is not configured on this rank.")
        if self._sft_active_envelope is not None:
            raise RuntimeError("Cannot checkpoint an uncommitted SFT batch.")
        return self._sft_loader.state_dict()

    def close_sft_dataloader(self) -> None:
        """Clear local batch state and release the loader reference."""
        self.abort_sft_batch()
        self._sft_loader_iterator = None
        self._sft_loader = None

    def _require_active_envelope(self) -> StepEnvelope:
        if self._sft_active_envelope is None:
            raise RuntimeError("There is no active SFT batch to commit.")
        return self._sft_active_envelope

    @staticmethod
    def _source_ids(batch: Mapping[str, Any], *, batch_size: int) -> tuple[str, ...]:
        for key in ("source_ids", "sample_keys"):
            values = batch.get(key)
            if isinstance(values, (list, tuple)) and len(values) == batch_size:
                return tuple(
                    str(source_id)
                    for value in values
                    for source_id in (
                        value if isinstance(value, (list, tuple)) else [value]
                    )
                )
        return tuple(f"unknown:{row}" for row in range(batch_size))

    @staticmethod
    def _source_tags(
        batch: Mapping[str, Any], *, batch_size: int
    ) -> list[dict[str, Any]]:
        values = batch.get("source_ids")
        if not isinstance(values, (list, tuple)) or len(values) != batch_size:
            return [{"source_id": f"unknown:{row}"} for row in range(batch_size)]
        return [
            (
                {"source_ids": [str(source_id) for source_id in value]}
                if isinstance(value, (list, tuple))
                else {"source_id": str(value)}
            )
            for value in values
        ]

    @staticmethod
    def _packing_metadata(batch: Mapping[str, Any]) -> dict[str, Any]:
        source_ids = batch["source_ids"]
        cu_seqlens = batch["cu_seqlens"]
        cu_seqlens_padded = batch["cu_seqlens_padded"]
        pack_lengths = [int(value) for value in batch["input_lengths"].tolist()]
        capacities = {int(value) for value in batch["pack_capacity"].tolist()}
        schema_versions = {
            int(value) for value in batch["packed_schema_version"].tolist()
        }
        if len(capacities) != 1 or len(schema_versions) != 1:
            raise ValueError("One Energon batch must use one packing schema and capacity.")
        return {
            "schema_version": schema_versions.pop(),
            "pack_count": len(pack_lengths),
            "source_count": sum(len(ids) for ids in source_ids),
            "source_counts": [len(ids) for ids in source_ids],
            "pack_lengths": pack_lengths,
            "pack_capacity": capacities.pop(),
            "boundaries": [
                {
                    "cu_seqlens": boundaries.tolist(),
                    "cu_seqlens_padded": padded_boundaries.tolist(),
                }
                for boundaries, padded_boundaries in zip(
                    cu_seqlens, cu_seqlens_padded
                )
            ],
        }

    @classmethod
    def _field_fingerprints(cls, batch: Mapping[str, Any]) -> dict[str, Any]:
        fingerprints: dict[str, Any] = {}
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                fingerprints[name] = {
                    "kind": "tensor",
                    "dtype": str(value.dtype),
                    "shape": tuple(value.shape),
                    "hash": cls._tensor_hash(value),
                }
            elif isinstance(value, PackedTensor):
                tensors = [tensor for tensor in value.tensors if tensor is not None]
                fingerprints[name] = {
                    "kind": "packed_tensor",
                    "rows": len(value),
                    "tensor_shapes": [tuple(tensor.shape) for tensor in tensors],
                    "tensor_hashes": [cls._tensor_hash(tensor) for tensor in tensors],
                }
        return fingerprints

    @staticmethod
    def _tensor_hash(tensor: torch.Tensor) -> str:
        value = tensor.detach().cpu().contiguous()
        return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


__all__ = ["SFTMegatronPolicyWorker", "StepEnvelope"]
