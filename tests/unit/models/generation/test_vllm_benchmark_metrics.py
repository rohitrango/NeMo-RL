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

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.generation.vllm.benchmark_metrics import (
    HistogramSnapshot,
    VllmBenchmarkSnapshot,
    merge_vllm_benchmark_snapshots,
    subtract_vllm_benchmark_snapshots,
    summarize_vllm_benchmark_metrics,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


def _histogram(count: int, total: float, buckets: dict[str, int]) -> HistogramSnapshot:
    return {"count": count, "sum": total, "buckets": buckets}


def test_vllm_benchmark_metrics_cover_one_rollout_window():
    baseline: VllmBenchmarkSnapshot = {
        "counters": {
            "vllm:generation_tokens": 100,
            "vllm:request_success": 2,
        },
        "histograms": {
            "vllm:request_prompt_tokens": _histogram(
                2, 30, {"10.0": 1, "20.0": 2, "+Inf": 2}
            ),
            "vllm:request_generation_tokens": _histogram(2, 20, {"10.0": 2, "+Inf": 2}),
            "vllm:time_to_first_token_seconds": _histogram(
                2, 0.3, {"0.1": 1, "0.2": 2, "+Inf": 2}
            ),
        },
    }
    current: VllmBenchmarkSnapshot = {
        "counters": {
            "vllm:generation_tokens": 150,
            "vllm:request_success": 6,
        },
        "histograms": {
            "vllm:request_prompt_tokens": _histogram(
                6, 110, {"10.0": 2, "20.0": 5, "+Inf": 6}
            ),
            "vllm:request_generation_tokens": _histogram(6, 70, {"10.0": 5, "+Inf": 6}),
            "vllm:time_to_first_token_seconds": _histogram(
                6, 1.1, {"0.1": 2, "0.2": 6, "+Inf": 6}
            ),
        },
    }

    delta = subtract_vllm_benchmark_snapshots(current, baseline)
    summary = summarize_vllm_benchmark_metrics(delta, elapsed_s=2.0)

    assert summary["completed_requests"] == 4
    assert summary["request_throughput_per_s"] == 2
    assert summary["input_tokens"] == 80
    assert summary["output_tokens"] == 50
    assert summary["input_token_throughput_per_s"] == 40
    assert summary["output_token_throughput_per_s"] == 25
    assert summary["mean_ttft_ms"] == pytest.approx(200)
    assert summary["median_ttft_ms_upper_bound"] == 200
    assert summary["p99_ttft_ms_upper_bound"] == 200


def test_vllm_benchmark_metrics_merge_data_parallel_engines():
    first: VllmBenchmarkSnapshot = {
        "counters": {"vllm:request_success": 2},
        "histograms": {
            "vllm:e2e_request_latency_seconds": _histogram(
                2, 3.0, {"1.0": 1, "2.0": 2, "+Inf": 2}
            )
        },
    }
    second: VllmBenchmarkSnapshot = {
        "counters": {"vllm:request_success": 3},
        "histograms": {
            "vllm:e2e_request_latency_seconds": _histogram(
                3, 6.0, {"1.0": 1, "2.0": 3, "+Inf": 3}
            )
        },
    }

    summary = summarize_vllm_benchmark_metrics(
        merge_vllm_benchmark_snapshots([first, second]), elapsed_s=5.0
    )

    assert summary["completed_requests"] == 5
    assert summary["request_throughput_per_s"] == 1
    assert summary["mean_e2e_latency_ms"] == 1800
    assert summary["median_e2e_latency_ms_upper_bound"] == 2000


def test_vllm_generation_preserves_metrics_for_each_dp_engine():
    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = cast(
        Any,
        {
            "vllm_cfg": {
                "enable_vllm_metrics_logger": True,
                "async_engine": True,
            }
        },
    )
    generation.worker_group = MagicMock()
    generation.weight_synchronizer = None
    generation.worker_group.dp_size = 2
    generation.worker_group.get_dp_leader_worker_idx.side_effect = lambda dp_idx: dp_idx
    generation.worker_group.run_single_worker_single_data.side_effect = (
        lambda _method_name, worker_idx: worker_idx
    )

    def metrics_for_engine(requests: int, tokens: int) -> VllmBenchmarkSnapshot:
        return {
            "counters": {},
            "histograms": {
                "vllm:request_generation_tokens": _histogram(
                    requests,
                    tokens,
                    {"100.0": requests, "+Inf": requests},
                ),
                "vllm:e2e_request_latency_seconds": _histogram(
                    requests,
                    float(requests),
                    {"1.0": requests, "+Inf": requests},
                ),
            },
        }

    worker_results = [
        {
            "vllm_benchmark_delta": metrics_for_engine(2, 10),
            "vllm_benchmark_elapsed_s": 2.0,
        },
        {
            "vllm_benchmark_delta": metrics_for_engine(4, 40),
            "vllm_benchmark_elapsed_s": 4.0,
        },
    ]
    with patch(
        "nemo_rl.models.generation.vllm.vllm_generation.ray.get",
        return_value=worker_results,
    ):
        metrics = generation.get_vllm_logger_metrics()

    assert metrics["vllm_benchmark"]["completed_requests"] == 6
    assert metrics["vllm_benchmark"]["request_throughput_per_s"] == 1.5
    assert metrics["vllm_benchmark_per_dp"][0]["completed_requests"] == 2
    assert metrics["vllm_benchmark_per_dp"][0]["request_throughput_per_s"] == 1
    assert metrics["vllm_benchmark_per_dp"][0]["output_token_throughput_per_s"] == 5
    assert metrics["vllm_benchmark_per_dp"][1]["completed_requests"] == 4
    assert metrics["vllm_benchmark_per_dp"][1]["request_throughput_per_s"] == 1
    assert metrics["vllm_benchmark_per_dp"][1]["output_token_throughput_per_s"] == 10
