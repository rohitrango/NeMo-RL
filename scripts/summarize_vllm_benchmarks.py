#!/usr/bin/env python3
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

"""Summarize vLLM benchmark metrics from recent local experiment runs.

Example:
    uv run python scripts/summarize_vllm_benchmarks.py \
        workspace/results/my-experiment --csv --all-runs
"""

import argparse
import csv
import json
import re
import statistics
import struct
import sys
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from wandb.proto import wandb_internal_pb2


METRIC_PREFIX = "generation_metrics/vllm_benchmark/"
WANDB_FILE_HEADER = struct.Struct("<4sHB")
WANDB_RECORD_HEADER = struct.Struct("<IHB")
WANDB_HEADER_IDENT = b":W&B"
WANDB_HEADER_MAGIC = 0xBEE1
WANDB_HEADER_VERSION = 0
LEVELDB_BLOCK_SIZE = 32768
RECORD_FULL = 1
RECORD_FIRST = 2
RECORD_MIDDLE = 3
RECORD_LAST = 4
VALID_RECORD_TYPES = {RECORD_FULL, RECORD_FIRST, RECORD_MIDDLE, RECORD_LAST}
REQUIRED_METRICS = (
    "completed_requests",
    "input_tokens",
    "measurement_window_s",
    "mean_decode_time_ms",
    "mean_e2e_latency_ms",
    "mean_itl_ms",
    "mean_prefill_time_ms",
    "mean_queue_time_ms",
    "mean_tpot_ms",
    "mean_ttft_ms",
    "output_tokens",
)
P99_METRICS = (
    "p99_queue_time_ms_upper_bound",
    "p99_prefill_time_ms_upper_bound",
    "p99_ttft_ms_upper_bound",
    "p99_tpot_ms_upper_bound",
    "p99_itl_ms_upper_bound",
    "p99_e2e_latency_ms_upper_bound",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing exp_XXX subdirectories.",
    )
    parser.add_argument(
        "num_runs",
        type=int,
        nargs="?",
        help="Maximum number of recent runs to include (default: all).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Number of initial benchmark windows to exclude (default: 1).",
    )
    parser.add_argument(
        "--include-startup",
        action="store_true",
        help="Also print tables that include the excluded warmup windows.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print the report as CSV instead of Markdown.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include each experiment row before the Mean and Sample SD rows.",
    )
    return parser.parse_args()


def find_run_file(exp_dir: Path) -> Path:
    run_files = [path for path in exp_dir.rglob("run-*.wandb") if path.is_file()]
    if not run_files:
        raise ValueError(f"No local W&B transaction log found under {exp_dir}")
    return max(run_files, key=lambda path: (path.stat().st_mtime, path.name))


def read_physical_record(stream: BinaryIO, run_file: Path) -> tuple[int, bytes] | None:
    offset = stream.tell()
    header = stream.read(WANDB_RECORD_HEADER.size)
    if not header:
        return None
    if len(header) != WANDB_RECORD_HEADER.size:
        raise ValueError(f"Truncated record header at offset {offset} in {run_file}")

    checksum, data_length, record_type = WANDB_RECORD_HEADER.unpack(header)
    if record_type not in VALID_RECORD_TYPES:
        raise ValueError(
            f"Invalid record type {record_type} at offset {offset} in {run_file}"
        )
    data = stream.read(data_length)
    if len(data) != data_length:
        raise ValueError(f"Truncated record data at offset {offset} in {run_file}")

    type_checksum = zlib.crc32(bytes((record_type,))) & 0xFFFFFFFF
    computed_checksum = zlib.crc32(data, type_checksum) & 0xFFFFFFFF
    if checksum != computed_checksum:
        raise ValueError(f"Invalid record checksum at offset {offset} in {run_file}")
    return record_type, data


def iter_wandb_records(run_file: Path) -> Iterator[bytes]:
    """Read logical records from a local W&B transaction log."""
    with run_file.open("rb") as stream:
        header = stream.read(WANDB_FILE_HEADER.size)
        if len(header) != WANDB_FILE_HEADER.size:
            raise ValueError(f"Invalid W&B file header in {run_file}")
        ident, magic, version = WANDB_FILE_HEADER.unpack(header)
        if (
            ident != WANDB_HEADER_IDENT
            or magic != WANDB_HEADER_MAGIC
            or version != WANDB_HEADER_VERSION
        ):
            raise ValueError(f"Invalid W&B file header in {run_file}")

        while True:
            block_space = LEVELDB_BLOCK_SIZE - stream.tell() % LEVELDB_BLOCK_SIZE
            if block_space < WANDB_RECORD_HEADER.size:
                padding = stream.read(block_space)
                if padding != bytes(block_space):
                    raise ValueError(f"Invalid block padding in {run_file}")

            physical_record = read_physical_record(stream, run_file)
            if physical_record is None:
                return
            record_type, data = physical_record
            if record_type == RECORD_FULL:
                yield data
                continue
            if record_type != RECORD_FIRST:
                raise ValueError(
                    f"Expected first record fragment in {run_file}, got {record_type}"
                )

            fragments = [data]
            while True:
                physical_record = read_physical_record(stream, run_file)
                if physical_record is None:
                    raise ValueError(f"Truncated fragmented record in {run_file}")
                record_type, data = physical_record
                fragments.append(data)
                if record_type == RECORD_LAST:
                    yield b"".join(fragments)
                    break
                if record_type != RECORD_MIDDLE:
                    raise ValueError(
                        f"Expected middle or last record fragment in {run_file}, "
                        f"got {record_type}"
                    )


def read_benchmark_history(run_file: Path) -> tuple[list[dict[str, float]], float]:
    rows_by_step: dict[int, dict[str, float]] = {}
    end_to_end_time_s: float | None = None
    for data in iter_wandb_records(run_file):
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue

        history = {}
        for item in record.history.item:
            key = item.key or "/".join(item.nested_key)
            if key in {"_runtime", "_step"} or key.startswith(METRIC_PREFIX):
                history[key] = json.loads(item.value_json)

        if "_runtime" in history:
            runtime = float(history["_runtime"])
            end_to_end_time_s = max(end_to_end_time_s or runtime, runtime)

        if "_step" not in history or not any(
            key.startswith(METRIC_PREFIX) for key in history
        ):
            continue
        step = int(history.pop("_step"))
        rows_by_step.setdefault(step, {}).update(history)

    rows = [rows_by_step[step] for step in sorted(rows_by_step)]
    if not rows:
        raise ValueError(f"No {METRIC_PREFIX} history found in {run_file}")
    if end_to_end_time_s is None:
        raise ValueError(f"No W&B _runtime history found in {run_file}")

    required_keys = {METRIC_PREFIX + metric for metric in REQUIRED_METRICS}
    for index, row in enumerate(rows, start=1):
        missing = required_keys - row.keys()
        if missing:
            raise ValueError(
                f"Benchmark window {index} in {run_file} is missing: "
                + ", ".join(sorted(missing))
            )
    return rows, end_to_end_time_s


def metric(row: dict[str, float], name: str) -> float:
    return float(row[METRIC_PREFIX + name])


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    measurement_window_s = sum(metric(row, "measurement_window_s") for row in rows)
    completed_requests = sum(metric(row, "completed_requests") for row in rows)
    input_tokens = sum(metric(row, "input_tokens") for row in rows)
    output_tokens = sum(metric(row, "output_tokens") for row in rows)
    if measurement_window_s <= 0 or completed_requests <= 0:
        raise ValueError("Benchmark totals must have positive time and request counts")

    result = {
        "windows": float(len(rows)),
        "measurement_window_s": measurement_window_s,
        "completed_requests": completed_requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_throughput_per_s": completed_requests / measurement_window_s,
        "input_token_throughput_per_s": input_tokens / measurement_window_s,
        "output_token_throughput_per_s": output_tokens / measurement_window_s,
        "total_token_throughput_per_s": (input_tokens + output_tokens)
        / measurement_window_s,
        "input_tokens_per_request": input_tokens / completed_requests,
        "output_tokens_per_request": output_tokens / completed_requests,
    }
    for name in REQUIRED_METRICS:
        if not name.startswith("mean_"):
            continue
        result[name] = (
            sum(metric(row, name) * metric(row, "completed_requests") for row in rows)
            / completed_requests
        )
    for name in P99_METRICS:
        values = [metric(row, name) for row in rows if METRIC_PREFIX + name in row]
        if values:
            result[name] = max(values)
    return result


def format_value(value: float | None, decimals: int) -> str:
    if value is None:
        return "—"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def print_table(headers: list[str], rows: list[list[str]], *, csv_output: bool) -> None:
    if csv_output:
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                ["" if value == "—" else value.replace(",", "") for value in row]
            )
        print()
        return

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    print(render(headers))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows:
        print(render(row))


def print_title(title: str, *, csv_output: bool) -> None:
    if csv_output:
        csv.writer(sys.stdout, lineterminator="\n").writerow([title])
    else:
        print(f"\n{title}")


def summary_rows(
    aggregates: dict[str, dict[str, float]],
    columns: list[tuple[str, str, int]],
    *,
    include_all_runs: bool,
) -> list[list[str]]:
    rows = []
    if include_all_runs:
        for run_name, values in aggregates.items():
            rows.append(
                [run_name]
                + [format_value(values[key], decimals) for _, key, decimals in columns]
            )

    means = {
        key: statistics.fmean(values[key] for values in aggregates.values())
        for _, key, _ in columns
    }
    rows.append(
        ["Mean"]
        + [format_value(means[key], max(decimals, 1)) for _, key, decimals in columns]
    )
    if len(aggregates) > 1:
        deviations = {
            key: statistics.stdev(values[key] for values in aggregates.values())
            for _, key, _ in columns
        }
    else:
        deviations = {key: None for _, key, _ in columns}
    rows.append(
        ["Sample SD"]
        + [
            format_value(deviations[key], max(decimals, 1))
            for _, key, decimals in columns
        ]
    )
    return rows


def print_report(
    title: str,
    histories: dict[str, list[dict[str, float]]],
    *,
    csv_output: bool,
    include_all_runs: bool,
) -> None:
    aggregates = {name: aggregate(rows) for name, rows in histories.items()}
    throughput_columns = [
        ("Windows", "windows", 0),
        ("Requests", "completed_requests", 0),
        ("Window (s)", "measurement_window_s", 1),
        ("Req/s", "request_throughput_per_s", 3),
        ("Input tok/s", "input_token_throughput_per_s", 1),
        ("Output tok/s", "output_token_throughput_per_s", 1),
        ("Total tok/s", "total_token_throughput_per_s", 1),
        ("Input/req", "input_tokens_per_request", 1),
        ("Output/req", "output_tokens_per_request", 1),
    ]
    latency_columns = [
        ("Queue ms", "mean_queue_time_ms", 1),
        ("Prefill ms", "mean_prefill_time_ms", 1),
        ("TTFT ms", "mean_ttft_ms", 1),
        ("TPOT ms", "mean_tpot_ms", 2),
        ("ITL ms", "mean_itl_ms", 2),
        ("Decode ms", "mean_decode_time_ms", 1),
        ("E2E ms", "mean_e2e_latency_ms", 1),
    ]

    throughput_headers = ["Run"] + [label for label, _, _ in throughput_columns]
    throughput_rows = summary_rows(
        aggregates, throughput_columns, include_all_runs=include_all_runs
    )
    print_title(f"{title} — throughput", csv_output=csv_output)
    print_table(throughput_headers, throughput_rows, csv_output=csv_output)

    latency_headers = ["Run"] + [label for label, _, _ in latency_columns]
    latency_rows = summary_rows(
        aggregates, latency_columns, include_all_runs=include_all_runs
    )
    print_title(f"{title} — request-weighted mean latency", csv_output=csv_output)
    print_table(latency_headers, latency_rows, csv_output=csv_output)

    if all(name in values for values in aggregates.values() for name in P99_METRICS):
        p99_columns = [
            ("Queue ms", "p99_queue_time_ms_upper_bound", 0),
            ("Prefill ms", "p99_prefill_time_ms_upper_bound", 0),
            ("TTFT ms", "p99_ttft_ms_upper_bound", 0),
            ("TPOT ms", "p99_tpot_ms_upper_bound", 0),
            ("ITL ms", "p99_itl_ms_upper_bound", 0),
            ("E2E ms", "p99_e2e_latency_ms_upper_bound", 0),
        ]
        p99_headers = ["Run"] + [label for label, _, _ in p99_columns]
        p99_rows = summary_rows(
            aggregates, p99_columns, include_all_runs=include_all_runs
        )
        print_title(
            f"{title} — worst observed p99 histogram upper bound",
            csv_output=csv_output,
        )
        print_table(p99_headers, p99_rows, csv_output=csv_output)


def print_end_to_end_report(
    end_to_end_times_s: dict[str, float],
    *,
    csv_output: bool,
    include_all_runs: bool,
) -> None:
    aggregates = {
        run_name: {
            "end_to_end_time_s": seconds,
            "end_to_end_time_min": seconds / 60,
        }
        for run_name, seconds in end_to_end_times_s.items()
    }
    columns = [
        ("End-to-end (s)", "end_to_end_time_s", 1),
        ("End-to-end (min)", "end_to_end_time_min", 2),
    ]
    headers = ["Run"] + [label for label, _, _ in columns]
    rows = summary_rows(aggregates, columns, include_all_runs=include_all_runs)
    print_title(
        "End-to-end workload time — W&B runtime through the final logged step "
        "(includes setup, generation, and policy training)",
        csv_output=csv_output,
    )
    print_table(headers, rows, csv_output=csv_output)


def main() -> None:
    args = parse_args()
    if args.num_runs is not None and args.num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")

    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Directory does not exist: {root}")

    exp_pattern = re.compile(r"exp_\d+")
    exp_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and exp_pattern.fullmatch(path.name)
    ]
    exp_dirs.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    selected = (
        exp_dirs
        if args.num_runs is None
        else exp_dirs[: min(args.num_runs, len(exp_dirs))]
    )
    if not selected:
        raise ValueError(f"No exp_XXX directories found under {root}")

    histories: dict[str, list[dict[str, float]]] = {}
    end_to_end_times_s: dict[str, float] = {}
    for exp_dir in selected:
        history, end_to_end_time_s = read_benchmark_history(find_run_file(exp_dir))
        if len(history) <= args.warmup_steps:
            raise ValueError(
                f"{exp_dir} has {len(history)} benchmark window(s), which is not "
                f"more than --warmup-steps={args.warmup_steps}"
            )
        histories[exp_dir.name] = history
        end_to_end_times_s[exp_dir.name] = end_to_end_time_s

    if not args.csv:
        print(f"Directory: {root}")
        print(
            f"Selected {len(selected)} of {len(exp_dirs)} experiment directories "
            "by directory modification time (newest first): "
            + ", ".join(path.name for path in selected)
        )
    print_end_to_end_report(
        end_to_end_times_s,
        csv_output=args.csv,
        include_all_runs=args.all_runs,
    )
    steady_histories = {
        name: rows[args.warmup_steps :] for name, rows in histories.items()
    }
    print_report(
        f"Steady state (excluded first {args.warmup_steps} window(s))",
        steady_histories,
        csv_output=args.csv,
        include_all_runs=args.all_runs,
    )
    if args.include_startup and args.warmup_steps:
        print_report(
            "All windows (including startup)",
            histories,
            csv_output=args.csv,
            include_all_runs=args.all_runs,
        )


if __name__ == "__main__":
    main()
