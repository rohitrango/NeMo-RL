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
        # --- BEGIN MEMRAY_ALL (diagnostic: tracks EVERY rank) ---
        import os as _mr_os

        self._mr_tracker = None
        self._mr_stopped = False
        self._mr_steps = 0
        if _mr_os.environ.get("NRL_MEMRAY") == "1":
            try:
                import memray as _mr

                _mr_dir = "/mnt/rl-workspace/rohitkumarj/rssprof/allranks"
                _mr_os.makedirs(_mr_dir, exist_ok=True)
                _mr_path = f"{_mr_dir}/memray_pid{_mr_os.getpid()}_{_mr_os.uname().nodename[-5:]}.bin"
                self._mr_tracker = _mr.Tracker(
                    _mr_path, native_traces=False, follow_fork=False
                )
                self._mr_tracker.__enter__()
                print(f"[MEMRAY_ALL] started -> {_mr_path}", flush=True)

                # Stop on a timer: non-leader ranks never call commit_sft_batch,
                # so a per-step hook would only fire on the DP leader. A timer
                # gives every rank a clean __exit__ and a flushed file.
                import threading as _mr_th

                def _mr_timer_stop() -> None:
                    import time as _mr_time

                    _mr_time.sleep(
                        float(_mr_os.environ.get("NRL_MEMRAY_SECONDS", "600"))
                    )
                    try:
                        if self._mr_tracker is not None and not self._mr_stopped:
                            self._mr_tracker.__exit__(None, None, None)
                            self._mr_stopped = True
                            self._mr_tracker = None
                            print("[MEMRAY_ALL] STOPPED (timer); flushed", flush=True)
                    except Exception as _e:  # noqa: BLE001
                        print(f"[MEMRAY_ALL] timer stop failed: {_e}", flush=True)

                _mr_th.Thread(target=_mr_timer_stop, daemon=True).start()
            except Exception as _mr_e:  # noqa: BLE001
                print(f"[MEMRAY_ALL] start failed: {_mr_e}", flush=True)
                self._mr_tracker = None
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

        # --- BEGIN LEAKPROBE (diagnostic; remove after investigation) ---
        import weakref as _lp_wr

        self._lp_fields_ref = None
        self._lp_tensor_ref = None
        self._lp_tensor_gib = 0.0
        try:
            self._lp_fields_ref = _lp_wr.ref(fields)
            _lp_big = None
            for _lp_k in list(fields.keys()):
                _lp_v = fields.get(_lp_k)
                if hasattr(_lp_v, "nelement") and hasattr(_lp_v, "element_size"):
                    _lp_nb = _lp_v.element_size() * _lp_v.nelement()
                    if _lp_big is None or _lp_nb > _lp_big[1]:
                        _lp_big = (_lp_v, _lp_nb)
            if _lp_big is not None:
                self._lp_tensor_ref = _lp_wr.ref(_lp_big[0])
                self._lp_tensor_gib = _lp_big[1] / 2**30
        except Exception as _lp_e:  # noqa: BLE001
            print(f"[LEAKPROBE] weakref setup failed: {_lp_e}", flush=True)
        # --- END LEAKPROBE ---
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

    def _mr_maybe_stop(self) -> None:
        """Stop the all-rank memray tracker after NRL_MEMRAY_STEPS steps."""
        import os as _mr_os2

        if getattr(self, "_mr_tracker", None) is None or self._mr_stopped:
            return
        self._mr_steps += 1
        if self._mr_steps >= int(_mr_os2.environ.get("NRL_MEMRAY_STEPS", "10")):
            try:
                self._mr_tracker.__exit__(None, None, None)
                self._mr_stopped = True
                self._mr_tracker = None
                print(f"[MEMRAY_ALL] STOPPED after {self._mr_steps} steps", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[MEMRAY_ALL] stop failed: {_e}", flush=True)

    def commit_sft_batch(self) -> None:
        """Release the active process-local batch after a successful step."""
        envelope = self._require_active_envelope()
        self._require_dp_client().clear_samples(
            sample_ids=envelope.meta.sample_ids,
            partition_id=envelope.meta.partition_id,
        )
        self._sft_active_envelope = None

        # --- BEGIN MEMRAY stop ---
        try:
            import os as _mr_os2

            _mr_n = int(_mr_os2.environ.get("NRL_MEMRAY_STEPS", "10"))
            if (
                getattr(self, "_mr_tracker", None) is not None
                and not self._mr_stopped
                and self._sft_next_batch_index >= _mr_n
            ):
                self._mr_tracker.__exit__(None, None, None)
                self._mr_stopped = True
                self._mr_tracker = None
                print(
                    f"[MEMRAY] tracking STOPPED after {self._sft_next_batch_index} "
                    "batches; file flushed",
                    flush=True,
                )
        except Exception as _mr_e2:  # noqa: BLE001
            print(f"[MEMRAY] stop failed: {_mr_e2}", flush=True)
        # --- END MEMRAY stop ---

        # --- BEGIN LEAKPROBE ---
        try:
            import gc as _lp_gc
            import os as _lp_os

            _lp_pg = _lp_os.sysconf("SC_PAGE_SIZE")

            def _lp_rss():
                return int(open("/proc/self/statm").read().split()[1]) * _lp_pg / 2**30

            _lp_r0 = _lp_rss()
            _lp_n = _lp_gc.collect()
            _lp_r1 = _lp_rss()
            _lp_fa = (
                self._lp_fields_ref() is not None if self._lp_fields_ref else None
            )
            _lp_ta = (
                self._lp_tensor_ref() is not None if self._lp_tensor_ref else None
            )
            # --- MALLOCTRIM: is the growth glibc arena fragmentation? ---
            # If RSS drops after malloc_trim(0), the memory was already free()d
            # but retained in glibc's per-thread arenas rather than returned to
            # the OS. That is fragmentation, not a leak, and the fix is
            # MALLOC_ARENA_MAX / periodic trim -- not chasing references.
            try:
                import ctypes as _mt_ctypes

                _mt_libc = _mt_ctypes.CDLL("libc.so.6")
                _mt_pg = _lp_os.sysconf("SC_PAGE_SIZE")

                def _mt_rss():
                    return (
                        int(open("/proc/self/statm").read().split()[1]) * _mt_pg / 2**30
                    )

                _mt_before = _mt_rss()
                _mt_rc = _mt_libc.malloc_trim(0)
                _mt_after = _mt_rss()
                _mt_thr = len(_lp_os.listdir("/proc/self/task"))
                print(
                    f"[MALLOCTRIM] batch={self._sft_next_batch_index} "
                    f"rss_before={_mt_before:.2f}Gi rss_after={_mt_after:.2f}Gi "
                    f"RECLAIMED={_mt_before - _mt_after:.2f}Gi rc={_mt_rc} "
                    f"threads={_mt_thr} "
                    f"arena_max={_lp_os.environ.get('MALLOC_ARENA_MAX', 'unset')}",
                    flush=True,
                )
            except Exception as _mt_e:  # noqa: BLE001
                print(f"[MALLOCTRIM] failed: {_mt_e}", flush=True)

            print(
                f"[LEAKPROBE] batch={self._sft_next_batch_index} "
                f"gc_collected={_lp_n} fields_alive={_lp_fa} tensor_alive={_lp_ta} "
                f"probe_tensor={self._lp_tensor_gib:.2f}Gi "
                f"rss_before={_lp_r0:.1f}Gi rss_after={_lp_r1:.1f}Gi "
                f"freed={_lp_r0 - _lp_r1:.2f}Gi",
                flush=True,
            )
            # --- Probe A: torch device + PINNED-HOST allocator accounting ---
            import torch as _lp_t

            _lp_dev_a = _lp_t.cuda.memory_allocated() / 2**30
            _lp_dev_r = _lp_t.cuda.memory_reserved() / 2**30
            try:
                _lp_hs = _lp_t.cuda.host_memory_stats()
                _lp_ha = _lp_hs.get("allocated_bytes.all.current", 0) / 2**30
                _lp_hr = _lp_hs.get("reserved_bytes.all.current", 0) / 2**30
                _lp_hp = _lp_hs.get("reserved_bytes.all.peak", 0) / 2**30
                _lp_hn = _lp_hs.get("num_host_alloc", -1)
            except Exception as _lp_he:  # noqa: BLE001
                _lp_ha = _lp_hr = _lp_hp = -1.0
                _lp_hn = -1
                print(f"[TORCHMEM] host_memory_stats failed: {_lp_he}", flush=True)
            print(
                f"[TORCHMEM] batch={self._sft_next_batch_index} "
                f"dev_alloc={_lp_dev_a:.2f}Gi dev_resv={_lp_dev_r:.2f}Gi "
                f"host_alloc={_lp_ha:.2f}Gi host_resv={_lp_hr:.2f}Gi "
                f"host_peak={_lp_hp:.2f}Gi host_nalloc={_lp_hn}",
                flush=True,
            )

            # --- Probe B: device allocation history + periodic snapshot ---
            _lp_snapdir = "/mnt/rl-workspace/rohitkumarj/rssprof"
            _lp_bi = self._sft_next_batch_index
            try:
                if _lp_bi == 1:
                    _lp_t.cuda.memory._record_memory_history(max_entries=100_000)
                    print("[TORCHMEM] _record_memory_history STARTED", flush=True)
                elif _lp_bi % 5 == 0:
                    _lp_f = f"{_lp_snapdir}/rank0_batch{_lp_bi}.pickle"
                    _lp_t.cuda.memory._dump_snapshot(_lp_f)
                    print(f"[TORCHMEM] snapshot -> {_lp_f}", flush=True)
            except Exception as _lp_se:  # noqa: BLE001
                print(f"[TORCHMEM] snapshot failed: {_lp_se}", flush=True)

            if _lp_fa:
                _lp_obj = self._lp_fields_ref()
                for _lp_ref in _lp_gc.get_referrers(_lp_obj)[:5]:
                    print(
                        f"[LEAKPROBE]   holder: {type(_lp_ref).__name__} "
                        f"{str(_lp_ref)[:150]}",
                        flush=True,
                    )
        except Exception as _lp_e:  # noqa: BLE001
            print(f"[LEAKPROBE] failed: {_lp_e}", flush=True)
        # --- END LEAKPROBE ---

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
