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

"""Check whether running the Energon SFT dataloader inside a Ray actor
changes `/dev/shm` growth, at `video_decode_thread_count=0` and `=8`.

Context: `logbook/energon_thread_ablation.md` root-caused the 16-node job's
`/dev/shm` exhaustion to a native (C-level) PyTorch reference leak in shared
tensor IPC, whose rate scales with decode throughput. That conclusion came
from CPU-only benchmarking with `num_workers=0` (`compare_threading_latency.py`,
no DataLoader worker fork at all) and from the real 16-node job (Ray actors,
`num_workers=2`, real forking). This script fills the gap between those two:
it builds the same production loader (`build_energon_sft_loader`, the exact
path `SFTMegatronPolicyWorker` uses) with the yaml's own `num_workers=2` so
DataLoader workers really fork -- but does it *inside a Ray actor*, on CPU
only, with no GPUs, no distributed training, and no other bottleneck.

Why this matters: `multiprocessing_context="fork"` workers are forked from
whichever process calls `iter(loader)`. In the real job that process is a
Ray actor -- already running Ray's own background threads (core worker
executor, heartbeat, log monitor, plasma object store client) before that
fork ever happens. POSIX `fork()` only preserves the calling thread; if any
of those Ray threads holds a C-level lock (glibc malloc arena lock, a
grpc/protobuf internal lock, etc.) at the instant of fork, the forked
DataLoader worker inherits a permanently-locked lock with no thread left to
release it. This is a different, and previously unchecked, hazard from the
`_ENERGON_AV_OPEN_LOCK` Python lock (already ruled out: it is only acquired
well after the loader's one-time fork, never held at fork time). A bare
Python process (`compare_threading_latency.py`) cannot surface this, since
it never runs inside Ray. This script isolates exactly that variable: same
loader, same config, same CPU-only setup, with Ray added.

Usage (run once per CPU pod, one thread-count arm per invocation):

    uv run --extra energon python tools/investigate_thread_decode_with_ray.py \\
        --video-decode-thread-count 0 --num-batches 300

    uv run --extra energon python tools/investigate_thread_decode_with_ray.py \\
        --video-decode-thread-count 8 --num-batches 300

Each invocation starts its own local (single-node) Ray instance, builds one
Ray actor, constructs the loader inside it, iterates batches, and samples
`/dev/shm` on the pod every `--shm-sample-interval-s` seconds throughout.
Report (samples + summary) is written to `--output` (JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = (
    REPO_ROOT
    / "examples"
    / "configs"
    / "sft_v2_tests"
    / "sft-super67b-generalist.n16soak.yaml"
)
DEFAULT_SOURCE_DATASET = Path(
    "/mnt/rl-workspace/rohitkumarj/data/subset_cook_v2/subset.yaml"
)
DEFAULT_VIDEO_ONLY_DATASET = (
    DEFAULT_SOURCE_DATASET.parent / "subset_video_leaves.yaml"
)
SHM_PATH = Path("/dev/shm")


# --------------------------------------------------------------------------
# Video-only dataset yaml (same logic as compare_threading_latency.py)
# --------------------------------------------------------------------------


def build_video_only_dataset_yaml(
    source_yaml: Path, out_yaml: Path, *, split: str = "train"
) -> Path:
    """Write a MetadatasetV2 yaml containing only the `video__*` leaves.

    Energon metadataset entries resolve `path` relative to the yaml's own
    directory, so `out_yaml` must live alongside (or under) `source_yaml`'s
    directory for the copied paths to keep resolving.
    """
    import yaml

    if out_yaml.exists():
        return out_yaml

    if out_yaml.parent.resolve() != source_yaml.parent.resolve():
        raise ValueError(
            f"{out_yaml} must live in {source_yaml.parent} so its leaf "
            "paths (relative to the yaml file) keep resolving."
        )

    with open(source_yaml) as f:
        source = yaml.safe_load(f)

    all_entries = source["splits"][split]["blend_epochized"]
    video_entries = [e for e in all_entries if e["path"].startswith("datasets/video__")]
    if not video_entries:
        raise ValueError(f"No 'datasets/video__*' leaves found in {source_yaml}.")

    video_only = {
        "__module__": source["__module__"],
        "__class__": source["__class__"],
        "splits": {
            split: {"blend_epochized": video_entries},
        },
    }
    with open(out_yaml, "w") as f:
        f.write(
            f"# Video-only subset of {source_yaml.name}, generated by "
            "investigate_thread_decode_with_ray.py.\n"
            "# Keep this file in the same directory as its source so leaf "
            "paths keep resolving.\n"
        )
        yaml.safe_dump(video_only, f, sort_keys=False)

    print(
        f"[investigate_thread_decode_with_ray] wrote {len(video_entries)}/"
        f"{len(all_entries)} video leaves -> {out_yaml}"
    )
    return out_yaml


# --------------------------------------------------------------------------
# /dev/shm monitor
# --------------------------------------------------------------------------


def _process_label(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm") as f:
            comm = f.read().strip()
    except OSError:
        comm = "?"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode(errors="replace").replace("\x00", " ").strip()
    except OSError:
        cmdline = ""
    label = f"{pid}:{comm}"
    if cmdline:
        label += f" ({cmdline[:100]})"
    return label


def scan_shm_fd_holders() -> dict[str, Any]:
    """Scan every process's open fds for handles into /dev/shm, including
    unlinked-but-still-open ones.

    Under the default `file_descriptor` torch sharing strategy, a shared
    tensor's backing segment is `shm_open`'d then immediately unlinked --
    it only shows up in `ls /dev/shm` for the brief window before unlink.
    After that, the only trace of it is a live file descriptor in whichever
    process still holds it, pointing at "/dev/shm/... (deleted)". This is
    the same technique the original leak investigation used (scanning
    /proc/<pid>/fd) to find that 1759/2000 sampled fds on the training
    job's rank-0 process pointed at deleted-but-still-open segments.
    `os.stat()` on the fd path still returns the real size and mtime even
    after unlink, since the inode stays alive as long as any fd references
    it -- that liveness is the leak.
    """
    now = time.time()
    per_process: dict[str, dict[str, Any]] = {}
    handles: list[dict[str, Any]] = []

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        fd_dir = f"/proc/{pid}/fd"
        try:
            fd_names = os.listdir(fd_dir)
        except OSError:
            continue  # process exited mid-scan, or not readable
        proc_label: str | None = None
        for fd in fd_names:
            fd_path = f"{fd_dir}/{fd}"
            try:
                target = os.readlink(fd_path)
            except OSError:
                continue
            if "/dev/shm/" not in target:
                continue
            try:
                st = os.stat(fd_path)
            except OSError:
                continue
            if proc_label is None:
                proc_label = _process_label(pid)
            bucket = per_process.setdefault(
                proc_label, {"pid": pid, "num_handles": 0, "total_bytes": 0}
            )
            bucket["num_handles"] += 1
            bucket["total_bytes"] += st.st_size
            handles.append(
                {
                    "pid": pid,
                    "process": proc_label,
                    "fd": fd,
                    "target": target,
                    "deleted": target.endswith("(deleted)"),
                    "size_bytes": st.st_size,
                    "age_s": now - st.st_mtime,
                }
            )

    handles.sort(key=lambda h: h["size_bytes"], reverse=True)
    return {
        "per_process": per_process,
        "top_handles": handles[:20],
        "total_handles": len(handles),
        "total_bytes": sum(h["size_bytes"] for h in handles),
    }


def sample_shm_usage() -> dict[str, Any]:
    usage = shutil.disk_usage(SHM_PATH)
    try:
        num_files = sum(1 for _ in SHM_PATH.iterdir())
    except OSError:
        num_files = -1
    fd_scan = scan_shm_fd_holders()
    return {
        "ts": time.time(),
        "used_bytes": usage.used,
        "total_bytes": usage.total,
        "pct": (usage.used / usage.total * 100.0) if usage.total else 0.0,
        "num_files": num_files,  # named (not-yet-unlinked) entries only
        "fd_scan": fd_scan,
    }


class ShmMonitor:
    """Background thread sampling `/dev/shm` on this pod at a fixed interval."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.time()
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = sample_shm_usage()
            self.samples.append(sample)
            elapsed = sample["ts"] - (self._t0 or sample["ts"])
            top_procs = sorted(
                sample["fd_scan"]["per_process"].items(),
                key=lambda kv: kv[1]["total_bytes"],
                reverse=True,
            )[:3]
            top_str = ", ".join(
                f"{label.split(' ', 1)[0]}={info['total_bytes'] / 1e9:.2f}GB/"
                f"{info['num_handles']}fd"
                for label, info in top_procs
            )
            print(
                f"[shm] t={elapsed:6.0f}s used={sample['used_bytes'] / 1e9:6.2f}GB "
                f"({sample['pct']:5.1f}%) named_files={sample['num_files']} "
                f"fd_total={sample['fd_scan']['total_bytes'] / 1e9:.2f}GB/"
                f"{sample['fd_scan']['total_handles']}fds"
                + (f" top=[{top_str}]" if top_str else "")
            )
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s + 5.0)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"peak_pct": 0.0, "peak_used_bytes": 0, "peak_num_files": 0}
        peak_sample = max(self.samples, key=lambda s: s["used_bytes"])
        peak_processes = sorted(
            peak_sample["fd_scan"]["per_process"].items(),
            key=lambda kv: kv[1]["total_bytes"],
            reverse=True,
        )
        return {
            "peak_pct": max(s["pct"] for s in self.samples),
            "peak_used_bytes": max(s["used_bytes"] for s in self.samples),
            "peak_num_files": max(s["num_files"] for s in self.samples),
            "start_used_bytes": self.samples[0]["used_bytes"],
            "end_used_bytes": self.samples[-1]["used_bytes"],
            "duration_s": self.samples[-1]["ts"] - self.samples[0]["ts"],
            # Who held the most /dev/shm at the single highest-usage sample.
            "peak_sample_fd_totals_by_process": {
                label: info for label, info in peak_processes
            },
            "peak_sample_top_handles": peak_sample["fd_scan"]["top_handles"][:10],
        }


# --------------------------------------------------------------------------
# Ray actor: builds and iterates the real production loader
# --------------------------------------------------------------------------


def _build_and_run(
    *,
    config_path: str,
    dataset_yaml: str,
    video_decode_thread_count: int,
    num_batches: int,
    packing_buffer_size: int,
    num_workers: int | None,
    prefetch_factor: int | None,
) -> dict[str, Any]:
    """Runs inside the Ray actor process. Same loader-construction path as
    `compare_threading_latency.py::run_single_arm`, except `num_workers` is
    left at the yaml's own value (2) by default, so DataLoader workers
    really fork -- that fork, happening from inside a Ray actor process
    with Ray's own background threads already running, is the condition
    under test.
    """
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.sft_v2 import MasterConfig
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.energon.sft_dataloader import build_energon_sft_loader
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = load_config(Path(config_path))
    overrides = [
        f"data.train.path={dataset_yaml}",
        f"+data.energon.task_encoder.options.video_decode_thread_count="
        f"{video_decode_thread_count}",
        f"data.energon.task_encoder.packing.buffer_size={packing_buffer_size}",
    ]
    if num_workers is not None:
        overrides.append(f"data.energon.num_workers={num_workers}")
    if prefetch_factor is not None:
        overrides.append(f"data.energon.prefetch_factor={prefetch_factor}")
    config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    master_config = MasterConfig.model_validate(resolved)

    processor = get_tokenizer(master_config.policy["tokenizer"], get_processor=True)
    batch_size = master_config.policy["train_micro_batch_size"]
    max_sequence_length = master_config.policy["max_total_sequence_length"]

    build_t0 = time.perf_counter()
    train_loader = build_energon_sft_loader(
        data_config=master_config.data,
        source=master_config.data["train"],
        processor=processor,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        split_role="train",
        logical_rank=0,
        logical_world_size=1,
        placement_fingerprint="investigate-thread-decode-with-ray-dp1",
    )
    build_s = time.perf_counter() - build_t0

    batches_seen = 0
    t0 = time.perf_counter()
    it = iter(train_loader)
    for _ in range(num_batches):
        try:
            next(it)
        except StopIteration:
            break
        batches_seen += 1
    elapsed_s = time.perf_counter() - t0

    return {
        "video_decode_thread_count": video_decode_thread_count,
        "num_workers": master_config.data["energon"].num_workers,
        "prefetch_factor": master_config.data["energon"].prefetch_factor,
        "packing_buffer_size": packing_buffer_size,
        "loader_build_s": build_s,
        "requested_batches": num_batches,
        "batches_seen": batches_seen,
        "elapsed_s": elapsed_s,
        "mean_ms_per_batch": (elapsed_s / batches_seen * 1000.0) if batches_seen else 0.0,
    }


def _make_decode_actor():
    import ray

    @ray.remote
    class DecodeActor:
        def build_and_run(self, **kwargs: Any) -> dict[str, Any]:
            return _build_and_run(**kwargs)

    return DecodeActor


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--dataset-yaml", type=Path, default=DEFAULT_VIDEO_ONLY_DATASET)
    parser.add_argument(
        "--video-decode-thread-count",
        type=int,
        required=True,
        help="0 or 8 -- the arm this pod is running.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=150,
        help="Upper bound on batches to pull. tc=0 is slow; this is a cap, not a target.",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=1800.0,
        help="Hard wall-clock cap on the whole run (loader build + iteration).",
    )
    parser.add_argument(
        "--shm-abort-pct",
        type=float,
        default=90.0,
        help=(
            "If /dev/shm usage crosses this percent, cancel the actor and "
            "stop early instead of running the pod's shm mount to 100%%. "
            "This pod's /dev/shm is only 8GiB -- hitting 100%% wedges the "
            "pod for any other work on it, same failure mode as the real job."
        ),
    )
    parser.add_argument(
        "--packing-buffer-size",
        type=int,
        default=500,
        help="Override for data.energon.task_encoder.packing.buffer_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override data.energon.num_workers. Default: leave the yaml's value (2).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override data.energon.prefetch_factor. Default: leave the yaml's value (2).",
    )
    parser.add_argument("--shm-sample-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=16,
        help=(
            "Passed to ray.init(num_cpus=...). Pods here report the full "
            "host's CPU count via nproc, not the pod's cgroup limit -- "
            "default matches rohitkumarj-cpu-ablation-pod's actual limit."
        ),
    )
    parser.add_argument(
        "--ray-object-store-memory-bytes",
        type=int,
        default=200 * 1024 * 1024,
        help=(
            "Passed to ray.init(object_store_memory=...). Ray's plasma "
            "object store lives in /dev/shm and, uncapped, sizes itself off "
            "the host's full memory (also visible via /proc here, not the "
            "pod's cgroup limit) -- on an 8GiB /dev/shm mount that alone "
            "could dominate the experiment before any decoding happens. "
            "Kept small since this script doesn't move data through Ray's "
            "object store, only through the DataLoader's own IPC."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _cancel_and_get_real_error(ray_module: Any, future: Any, *, fallback: str) -> str:
    """Cancel the actor task and report its actual exception if one is
    already available, instead of a cancel-plumbing error masking it.

    `ray.cancel(..., force=True)` is rejected for plain (non-async, non-
    threaded) actor tasks in this Ray version -- irrelevant to what we
    actually want here: by the time --shm-abort-pct or --max-duration-
    seconds trips, the actor has very often already crashed with the real
    `RuntimeError: unable to allocate shared memory` on its own. Try a
    plain cancel (no force), then a short non-blocking ray.get() to surface
    that real exception if it is there; fall back to the threshold message
    only if the actor is genuinely still running.
    """
    try:
        ray_module.cancel(future)
    except Exception:  # noqa: BLE001 -- best-effort; the get() below is what matters
        pass
    try:
        ray_module.get(future, timeout=5.0)
    except ray_module.exceptions.GetTimeoutError:
        return fallback
    except Exception as exc:  # noqa: BLE001 -- this is the real crash reason
        return f"{fallback}; actor's real exception: {type(exc).__name__}: {exc}"
    return fallback


def main() -> None:
    args = parse_args()
    dataset_yaml = build_video_only_dataset_yaml(args.source_dataset, args.dataset_yaml)

    import ray

    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        num_cpus=args.ray_num_cpus,
        object_store_memory=args.ray_object_store_memory_bytes,
    )

    monitor = ShmMonitor(interval_s=args.shm_sample_interval_s)
    monitor.start()

    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        DecodeActor = _make_decode_actor()
        actor = DecodeActor.remote()
        future = actor.build_and_run.remote(
            config_path=str(args.config),
            dataset_yaml=str(dataset_yaml),
            video_decode_thread_count=args.video_decode_thread_count,
            num_batches=args.num_batches,
            packing_buffer_size=args.packing_buffer_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        start_t = time.time()
        # Poll instead of one blocking ray.get(): lets us cancel early if
        # /dev/shm crosses --shm-abort-pct, instead of running this pod's
        # small (8GiB) shm mount to 100% and wedging it, same failure mode
        # under test.
        while True:
            done, _ = ray.wait([future], timeout=2.0)
            if done:
                result = ray.get(done[0])
                break
            if monitor.samples and monitor.samples[-1]["pct"] >= args.shm_abort_pct:
                error = (
                    f"aborted: /dev/shm hit {monitor.samples[-1]['pct']:.1f}% "
                    f"(>= --shm-abort-pct={args.shm_abort_pct})"
                )
                print(f"[investigate_thread_decode_with_ray] {error}")
                error = _cancel_and_get_real_error(ray, future, fallback=error)
                break
            if time.time() - start_t >= args.max_duration_seconds:
                error = f"timed out after {args.max_duration_seconds}s"
                print(f"[investigate_thread_decode_with_ray] {error}")
                error = _cancel_and_get_real_error(ray, future, fallback=error)
                break
    except Exception as exc:  # noqa: BLE001 -- report whatever failed, don't lose shm samples
        error = f"{type(exc).__name__}: {exc}"
        print(f"[investigate_thread_decode_with_ray] actor failed: {error}")
    finally:
        monitor.stop()
        ray.shutdown()

    report = {
        "video_decode_thread_count": args.video_decode_thread_count,
        "config_path": str(args.config),
        "dataset_yaml": str(dataset_yaml),
        "packing_buffer_size": args.packing_buffer_size,
        "result": result,
        "error": error,
        "shm_summary": monitor.summary(),
        "shm_samples": monitor.samples,
    }
    out_path = args.output or Path(
        f"/tmp/investigate_thread_decode_with_ray_tc{args.video_decode_thread_count}.json"
    )
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[investigate_thread_decode_with_ray] wrote {out_path}")
    if result is not None:
        print(
            f"tc={args.video_decode_thread_count}: batches={result['batches_seen']} "
            f"elapsed={result['elapsed_s']:.1f}s "
            f"({result['mean_ms_per_batch']:.1f} ms/batch)"
        )
    print(
        f"shm peak: {report['shm_summary'].get('peak_pct', 0.0):.1f}% "
        f"({report['shm_summary'].get('peak_used_bytes', 0) / 1e9:.2f}GB), "
        f"files peak: {report['shm_summary'].get('peak_num_files', 0)}"
    )


if __name__ == "__main__":
    main()
