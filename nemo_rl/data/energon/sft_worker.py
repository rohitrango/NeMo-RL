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

import gc
import os as _ld_os
import threading as _ld_threading
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
            # Build the processor in-worker instead of receiving a pickled one.
            #
            # trust_remote_code processor classes (e.g. NemotronH_Omni) live in
            # transformers_modules.<repo>.<mod>, which transformers generates at
            # runtime under HF_MODULES_CACHE. Pickling such an object across the
            # Ray actor boundary stores it by qualified name, so every worker must
            # resolve that dynamically-created module before it can deserialise:
            #   ModuleNotFoundError: No module named
            #     'transformers_modules.<repo>.processing_nemotron_h_omni'
            # SLURM/SPMD never hits this because each rank constructs its own
            # processor; only the driver->actor hop makes it a pickle problem.
            #
            # self.cfg is the policy config (megatron_policy_worker.py:451) and
            # carries the same tokenizer block the driver used, so rebuilding here
            # is equivalent and needs nothing extra shipped over the wire.
            tok_cfg = (self.cfg or {}).get("tokenizer")
            if tok_cfg is None:
                raise ValueError(
                    "SFTv2 requires a multimodal processor on policy workers, and "
                    "policy.tokenizer was not available to build one locally."
                )
            from nemo_rl.algorithms.utils import get_tokenizer

            self._sft_processor = get_tokenizer(tok_cfg, get_processor=True)

        # Megatron's prepacked CP path requires every padded sub-sequence to be
        # divisible by 2 * cp_size. Nothing in EnergonPackingOptions ties the pad
        # multiple to CP, so a mismatch otherwise surfaces as a ValueError deep in
        # the first forward pass instead of here, where the fix is obvious.
        cp_size = parallel_state.get_context_parallel_world_size()
        # data_config["energon"] is a parsed EnergonLoaderConfig here but a plain
        # dict on other call paths, so walk it without assuming either shape.
        def _field(obj: Any, key: str) -> Any:
            if obj is None:
                return None
            if isinstance(obj, Mapping):
                return obj.get(key)
            return getattr(obj, key, None)

        pad_multiple = _field(
            _field(
                _field(_field(data_config, "energon"), "task_encoder"), "packing"
            ),
            "options",
        )
        pad_multiple = _field(pad_multiple, "sequence_length_pad_multiple")
        if cp_size > 1 and pad_multiple is not None:
            if pad_multiple % (2 * cp_size):
                raise ValueError(
                    "Energon packing sequence_length_pad_multiple "
                    f"({pad_multiple}) must be divisible by 2 * "
                    f"context_parallel_size ({2 * cp_size}); Megatron slices each "
                    "padded sub-sequence across CP ranks in two halves."
                )

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

    def _ld_mark(self, phase: str) -> None:
        """Record the phase now being entered, for the stall watchdog."""
        if getattr(self, "_ld_on", None) is None:
            self._ld_on = _ld_os.environ.get("NRL_LOADDIAG") == "1"
        if not self._ld_on:
            return
        now = time.monotonic()
        previous = getattr(self, "_ld_phase", None)
        if previous is not None:
            print(
                "[LOADDIAG] batch=%d %s done in %.3fs -> %s"
                % (
                    getattr(self, "_sft_next_batch_index", -1),
                    previous,
                    now - getattr(self, "_ld_t0", now),
                    phase,
                ),
                flush=True,
            )
        self._ld_phase = None if phase == "idle" else phase
        self._ld_t0 = now
        if getattr(self, "_ld_watchdog", None) is None:
            self._ld_watchdog = _ld_threading.Thread(
                target=self._ld_watch, name="sft-loaddiag", daemon=True
            )
            self._ld_watchdog.start()

    def _ld_watch(self) -> None:
        """Print the in-flight phase every 30s so a stall names itself."""
        while True:
            time.sleep(15)
            phase = getattr(self, "_ld_phase", None)
            if phase is None:
                continue
            elapsed = time.monotonic() - getattr(self, "_ld_t0", time.monotonic())
            if elapsed < 30:
                continue
            try:
                workers = sum(
                    1
                    for pid in _ld_os.listdir("/proc")
                    if pid.isdigit()
                    and "pt_data_worker"
                    in open("/proc/%s/comm" % pid, errors="ignore").read()
                )
            except Exception:  # noqa: BLE001
                workers = -1
            print(
                "[LOADDIAG] STUCK batch=%d phase=%s elapsed=%.0fs data_workers=%d"
                % (
                    getattr(self, "_sft_next_batch_index", -1),
                    phase,
                    elapsed,
                    workers,
                ),
                flush=True,
            )

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
        self._ld_mark("iter")
        batch = next(self._sft_loader_iterator)
        self._ld_mark("prepare")
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
        self._ld_mark("post-prepare")
        load_seconds = time.monotonic() - started
        batch_size = prepared.size
        source_ids = self._source_ids(prepared, batch_size=batch_size)
        partition_id = (
            f"sft_v2_dp{self._sft_logical_rank}_batch{self._sft_next_batch_index}"
        )
        sample_ids = [f"{partition_id}_row{row}" for row in range(batch_size)]
        self._ld_mark("tensordict")
        fields = local_batch_to_tensordict(prepared, batch_size=batch_size)

        self._ld_mark("publish")
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
            load_seconds=load_seconds,
            valid_tokens=valid_tokens,
        )
        # The controller blocks on this entire call, so load_seconds -- which
        # surfaces as loader_wait and gpu_idle_time -- has to span all of it.
        # Measured after prepare() alone it missed publish and envelope
        # construction, under-reporting real GPU idle by about half.
        envelope = replace(envelope, load_seconds=time.monotonic() - started)

        self._ld_mark("idle")
        # Keep a light copy so a failed step can dump what was actually fed in.
        # int64 [rows, seq] at 524288 is ~8MB each; two of them is acceptable
        # against the alternative of not knowing which data killed the run.
        try:
            self._sft_last_batch = {
                "batch_index": self._sft_next_batch_index,
                "input_ids": prepared["input_ids"].detach().to("cpu").clone(),
                "token_mask": prepared["token_mask"].detach().to("cpu").clone(),
                "input_lengths": prepared["input_lengths"].detach().to("cpu").clone(),
                "source_ids": source_ids,
                "source_lengths": prepared["source_lengths"],
            }
        except Exception:  # never let bookkeeping break the data path
            self._sft_last_batch = None

        self._sft_active_envelope = envelope
        self._sft_next_batch_index += 1
        return envelope

    def dump_failed_batch(self) -> str:
        """Decode and write the last batch handed to the trainer.

        Called by the controller after a step fails. Returns the path written,
        or a short reason if nothing could be dumped.
        """
        import json
        import os

        batch = getattr(self, "_sft_last_batch", None)
        if batch is None:
            return "no retained batch"
        try:
            out_dir = os.environ.get(
                "NRL_FAILED_BATCH_DIR", "/mnt/rl-workspace/rohitkumarj/failed_batches"
            )
            os.makedirs(out_dir, exist_ok=True)
            rank = self._sft_logical_rank
            path = os.path.join(
                out_dir, f"batch{batch['batch_index']:06d}_dp{rank}.json"
            )

            ids = batch["input_ids"]
            mask = batch["token_mask"]
            rows = []
            for row in range(ids.shape[0]):
                row_len = int(batch["input_lengths"][row])
                tok = ids[row][:row_len]
                msk = mask[row][:row_len]
                # decode mask-runs so the trained vs untrained split is visible
                spans = []
                start = 0
                for i in range(1, row_len + 1):
                    if i == row_len or bool(msk[i]) != bool(msk[start]):
                        chunk = [int(t) for t in tok[start:i]]
                        text = self.tokenizer.decode(chunk, skip_special_tokens=False)
                        spans.append(
                            {
                                "trained": bool(msk[start]),
                                "n_tokens": len(chunk),
                                "text": text[:400],
                            }
                        )
                        start = i
                rows.append(
                    {
                        "row": row,
                        "length": row_len,
                        "source_ids": list(batch["source_ids"][row]),
                        "source_lengths": [int(x) for x in batch["source_lengths"][row]],
                        "n_trained_tokens": int((msk != 0).sum()),
                        "min_token_id": int(tok.min()),
                        "max_token_id": int(tok.max()),
                        "spans": spans[:40],
                    }
                )

            with open(path, "w") as handle:
                json.dump(
                    {"batch_index": batch["batch_index"], "dp_rank": rank, "rows": rows},
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            return path
        except Exception as error:  # noqa: BLE001
            return f"dump failed: {error}"

    def commit_sft_batch(self) -> None:
        """Release the active process-local batch after a successful step."""
        envelope = self._require_active_envelope()
        self._require_dp_client().clear_samples(
            sample_ids=envelope.meta.sample_ids,
            partition_id=envelope.meta.partition_id,
        )
        self._sft_active_envelope = None

        # Reclaim the batch we just released before the loader produces the
        # next one. Without this the run stalls inside next() on the energon
        # iterator within a handful of steps -- the data workers stay alive but
        # stop yielding. This was previously happening only as a side effect of
        # the LEAKPROBE diagnostics; it is load-bearing, so it is explicit now.
        gc.collect()

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


__all__ = ["SFTMegatronPolicyWorker", "StepEnvelope"]
