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

"""Convert cumulative vLLM Prometheus metrics into rollout benchmark metrics."""

import math
from typing import TypedDict


VLLM_BENCHMARK_COUNTERS = frozenset(
    {
        "vllm:prompt_tokens",
        "vllm:generation_tokens",
        "vllm:request_success",
    }
)

VLLM_BENCHMARK_HISTOGRAMS = frozenset(
    {
        "vllm:request_prompt_tokens",
        "vllm:request_generation_tokens",
        "vllm:time_to_first_token_seconds",
        "vllm:request_time_per_output_token_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
    }
)


class HistogramSnapshot(TypedDict):
    count: int
    sum: float
    buckets: dict[str, int]


class VllmBenchmarkSnapshot(TypedDict):
    counters: dict[str, int]
    histograms: dict[str, HistogramSnapshot]


def empty_vllm_benchmark_snapshot() -> VllmBenchmarkSnapshot:
    return {"counters": {}, "histograms": {}}


def _counter_delta(current: int, baseline: int) -> int:
    # Treat a lower current value as a counter reset.
    return current - baseline if current >= baseline else current


def subtract_vllm_benchmark_snapshots(
    current: VllmBenchmarkSnapshot,
    baseline: VllmBenchmarkSnapshot,
) -> VllmBenchmarkSnapshot:
    """Subtract cumulative metric snapshots, accounting for counter resets."""
    delta = empty_vllm_benchmark_snapshot()
    for name, value in current["counters"].items():
        delta["counters"][name] = _counter_delta(
            value, baseline["counters"].get(name, 0)
        )

    for name, value in current["histograms"].items():
        baseline_value = baseline["histograms"].get(
            name, {"count": 0, "sum": 0.0, "buckets": {}}
        )
        reset = value["count"] < baseline_value["count"]
        if reset:
            delta["histograms"][name] = {
                "count": value["count"],
                "sum": value["sum"],
                "buckets": dict(value["buckets"]),
            }
            continue

        delta["histograms"][name] = {
            "count": value["count"] - baseline_value["count"],
            "sum": value["sum"] - baseline_value["sum"],
            "buckets": {
                bound: count - baseline_value["buckets"].get(bound, 0)
                for bound, count in value["buckets"].items()
            },
        }
    return delta


def merge_vllm_benchmark_snapshots(
    snapshots: list[VllmBenchmarkSnapshot],
) -> VllmBenchmarkSnapshot:
    """Sum metric deltas from all vLLM data-parallel engines."""
    merged = empty_vllm_benchmark_snapshot()
    for snapshot in snapshots:
        for name, value in snapshot["counters"].items():
            merged["counters"][name] = merged["counters"].get(name, 0) + value
        for name, value in snapshot["histograms"].items():
            target = merged["histograms"].setdefault(
                name, {"count": 0, "sum": 0.0, "buckets": {}}
            )
            target["count"] += value["count"]
            target["sum"] += value["sum"]
            for bound, count in value["buckets"].items():
                target["buckets"][bound] = target["buckets"].get(bound, 0) + count
    return merged


def _histogram_quantile_upper_bound(
    histogram: HistogramSnapshot, quantile: float
) -> float | None:
    if histogram["count"] <= 0:
        return None

    rank = math.ceil(histogram["count"] * quantile)
    finite_buckets = sorted(
        (float(bound), count)
        for bound, count in histogram["buckets"].items()
        if math.isfinite(float(bound))
    )
    for bound, cumulative_count in finite_buckets:
        if cumulative_count >= rank:
            return bound
    return None


def summarize_vllm_benchmark_metrics(
    snapshot: VllmBenchmarkSnapshot, elapsed_s: float
) -> dict[str, float]:
    """Create vllm-bench-style scalar metrics for one rollout window."""
    counters = snapshot["counters"]
    histograms = snapshot["histograms"]
    prompt_histogram = histograms.get("vllm:request_prompt_tokens")
    generation_histogram = histograms.get("vllm:request_generation_tokens")
    e2e_histogram = histograms.get("vllm:e2e_request_latency_seconds")

    prompt_tokens = int(
        prompt_histogram["sum"]
        if prompt_histogram is not None
        else counters.get("vllm:prompt_tokens", 0)
    )
    generation_tokens = int(
        generation_histogram["sum"]
        if generation_histogram is not None
        else counters.get("vllm:generation_tokens", 0)
    )
    completed_requests = (
        e2e_histogram["count"]
        if e2e_histogram is not None
        else counters.get("vllm:request_success", 0)
    )

    summary = {
        "measurement_window_s": elapsed_s,
        "completed_requests": float(completed_requests),
        "input_tokens": float(prompt_tokens),
        "output_tokens": float(generation_tokens),
    }
    if elapsed_s > 0:
        summary.update(
            {
                "request_throughput_per_s": completed_requests / elapsed_s,
                "input_token_throughput_per_s": prompt_tokens / elapsed_s,
                "output_token_throughput_per_s": generation_tokens / elapsed_s,
                "total_token_throughput_per_s": (prompt_tokens + generation_tokens)
                / elapsed_s,
            }
        )

    latency_histograms = {
        "ttft": "vllm:time_to_first_token_seconds",
        "tpot": "vllm:request_time_per_output_token_seconds",
        "itl": "vllm:inter_token_latency_seconds",
        "e2e_latency": "vllm:e2e_request_latency_seconds",
        "queue_time": "vllm:request_queue_time_seconds",
        "prefill_time": "vllm:request_prefill_time_seconds",
        "decode_time": "vllm:request_decode_time_seconds",
    }
    for short_name, metric_name in latency_histograms.items():
        histogram = histograms.get(metric_name)
        if histogram is None or histogram["count"] <= 0:
            continue
        summary[f"mean_{short_name}_ms"] = histogram["sum"] / histogram["count"] * 1000
        for label, quantile in (("median", 0.5), ("p99", 0.99)):
            upper_bound = _histogram_quantile_upper_bound(histogram, quantile)
            if upper_bound is not None:
                summary[f"{label}_{short_name}_ms_upper_bound"] = upper_bound * 1000
    return summary
