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

"""Benchmark NeMo-RL versus Energon packing and loader topologies."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.sft import MasterConfig, prepare_sft_batch
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.energon import build_energon_sft_loader
from nemo_rl.data.energon.multimodal.packing.sft import (
    pack_selected_samples,
    select_samples_to_pack,
)
from nemo_rl.data.energon.multimodal.types import EncodedSFTSample
from nemo_rl.data.packing import get_packer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile for an empty list.")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _timing_summary(seconds: list[float]) -> dict[str, float | int]:
    if not seconds:
        raise ValueError("At least one timing sample is required.")
    return {
        "measurements": len(seconds),
        "total_seconds": sum(seconds),
        "mean_seconds": statistics.fmean(seconds),
        "median_seconds": statistics.median(seconds),
        "p95_seconds": _percentile(seconds, 0.95),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
    }


def _log_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    records.append(record)
    print(json.dumps(record, indent=2, sort_keys=True))
    print("-----")


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _realistic_lengths(
    *, batch_size: int, capacity: int, seed: int
) -> list[int]:
    """Sample a long-tailed length profile based on the 131K blend pass."""
    rng = random.Random(seed)
    reference_capacity = 131_072
    reference_bands = (
        (0.50, 40, 2_054),
        (0.25, 2_055, 4_457),
        (0.15, 4_458, 12_371),
        (0.05, 12_372, 24_448),
        (0.04, 24_449, 88_157),
        (0.01, 88_158, reference_capacity),
    )
    lengths: list[int] = []
    for _ in range(batch_size):
        draw = rng.random()
        cumulative = 0.0
        for probability, reference_low, reference_high in reference_bands:
            cumulative += probability
            if draw <= cumulative:
                low = max(1, round(reference_low * capacity / reference_capacity))
                high = max(low, round(reference_high * capacity / reference_capacity))
                lengths.append(rng.randint(low, min(high, capacity)))
                break
    if len(lengths) != batch_size:
        raise RuntimeError("The synthetic length sampler returned too few values.")
    return lengths


def _synthetic_samples(
    lengths: list[int], *, rank: int
) -> list[EncodedSFTSample]:
    return [
        EncodedSFTSample(
            __key__=f"rank{rank}-sample{index}",
            __restore_key__=(f"rank{rank}-sample{index}",),
            message_log=[],
            length=length,
            packing_cost=length,
            loss_multiplier=1.0,
            group_key=("text",),
            sample_key=f"rank{rank}-sample{index}",
        )
        for index, length in enumerate(lengths)
    ]


def _time_call(call: Callable[[], Any], *, repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
    return timings


def _run_packing_ablation(
    *,
    dp_size: int,
    global_batch_sizes: list[int],
    max_sequence_lengths: list[int],
    algorithm: str,
    sequence_length_pad_multiple: int,
    warmup: int,
    repeats: int,
    seed: int,
    records: list[dict[str, Any]],
) -> None:
    for capacity in max_sequence_lengths:
        if sequence_length_pad_multiple > capacity:
            raise ValueError(
                "sequence_length_pad_multiple cannot exceed max sequence length."
            )
        for global_batch_size in global_batch_sizes:
            if global_batch_size % dp_size:
                raise ValueError(
                    f"Global batch size {global_batch_size} is not divisible by "
                    f"DP size {dp_size}."
                )
            lengths = _realistic_lengths(
                batch_size=global_batch_size,
                capacity=capacity,
                seed=seed + capacity + global_batch_size,
            )
            padded_lengths = [
                min(capacity, _round_up(length, sequence_length_pad_multiple))
                for length in lengths
            ]
            batch = BatchedDataDict(
                {
                    "input_lengths": torch.tensor(lengths, dtype=torch.int32),
                    "sample_ids": list(range(global_batch_size)),
                }
            )
            sequence_packing_args = {
                "algorithm": algorithm,
                "max_tokens_per_microbatch": capacity,
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": sequence_length_pad_multiple,
            }

            def nemo_rl_pack() -> Any:
                return batch.shard_by_batch_size(
                    shards=dp_size,
                    batch_size=global_batch_size,
                    sequence_packing_args=sequence_packing_args,
                )

            local_batch_size = global_batch_size // dp_size
            local_samples = [
                _synthetic_samples(
                    lengths[
                        rank * local_batch_size : (rank + 1) * local_batch_size
                    ],
                    rank=rank,
                )
                for rank in range(dp_size)
            ]
            local_packers = [get_packer(algorithm, capacity) for _ in range(dp_size)]

            def energon_pack_rank(rank: int) -> int:
                packs = select_samples_to_pack(
                    local_samples[rank],
                    packer=local_packers[rank],
                    sequence_length_pad_multiple=sequence_length_pad_multiple,
                )
                physical = [
                    pack_selected_samples(
                        selected,
                        max_sequence_length=capacity,
                        sequence_length_pad_multiple=sequence_length_pad_multiple,
                    )
                    for selected in packs
                ]
                return len(physical)

            for _ in range(warmup):
                nemo_rl_pack()
                for rank in range(dp_size):
                    energon_pack_rank(rank)

            nemo_rl_times = _time_call(nemo_rl_pack, repeats=repeats)
            energon_parallel_wall_times: list[float] = []
            energon_aggregate_cpu_times: list[float] = []
            for _ in range(repeats):
                rank_times: list[float] = []
                for rank in range(dp_size):
                    started = time.perf_counter()
                    energon_pack_rank(rank)
                    rank_times.append(time.perf_counter() - started)
                energon_parallel_wall_times.append(max(rank_times))
                energon_aggregate_cpu_times.append(sum(rank_times))

            central_packer = get_packer(
                algorithm,
                capacity,
                min_bin_count=dp_size,
                bin_count_multiple=dp_size,
            )
            central_bins = central_packer.pack(padded_lengths)
            local_bin_counts = [energon_pack_rank(rank) for rank in range(dp_size)]
            padded_token_count = sum(padded_lengths)
            nemo_rl_timing = _timing_summary(nemo_rl_times)
            energon_wall_timing = _timing_summary(energon_parallel_wall_times)
            _log_record(
                records,
                {
                    "benchmark": "packing",
                    "configuration": {
                        "algorithm": algorithm,
                        "dp_size": dp_size,
                        "global_batch_size": global_batch_size,
                        "local_batch_size": local_batch_size,
                        "max_sequence_length": capacity,
                        "sequence_length_pad_multiple": sequence_length_pad_multiple,
                        "warmup": warmup,
                        "repeats": repeats,
                        "seed": seed,
                    },
                    "lengths": {
                        "min": min(lengths),
                        "median": statistics.median(lengths),
                        "p90": _percentile([float(value) for value in lengths], 0.90),
                        "p95": _percentile([float(value) for value in lengths], 0.95),
                        "p99": _percentile([float(value) for value in lengths], 0.99),
                        "max": max(lengths),
                        "unpadded_tokens": sum(lengths),
                        "padded_tokens": padded_token_count,
                    },
                    "nemo_rl_central_pack_and_shard": {
                        "timing": nemo_rl_timing,
                        "physical_bins": len(central_bins),
                        "packing_efficiency": padded_token_count
                        / (len(central_bins) * capacity),
                        "sequences_per_second": global_batch_size
                        / float(nemo_rl_timing["mean_seconds"]),
                    },
                    "energon_rank_local_pack": {
                        "parallel_wall_timing_estimate": energon_wall_timing,
                        "aggregate_cpu_timing": _timing_summary(
                            energon_aggregate_cpu_times
                        ),
                        "physical_bins_by_rank": local_bin_counts,
                        "physical_bins": sum(local_bin_counts),
                        "packing_efficiency": padded_token_count
                        / (sum(local_bin_counts) * capacity),
                        "sequences_per_second_estimate": global_batch_size
                        / float(energon_wall_timing["mean_seconds"]),
                    },
                    "energon_over_nemo_rl_speedup_estimate": float(
                        nemo_rl_timing["mean_seconds"]
                    )
                    / float(energon_wall_timing["mean_seconds"]),
                },
            )


def _load_master_config(
    config_path: str, overrides: list[str]
) -> MasterConfig:
    register_omegaconf_resolvers()
    raw_config = load_config(config_path)
    if overrides:
        raw_config = parse_hydra_overrides(raw_config, overrides)
    return MasterConfig.model_validate(
        OmegaConf.to_container(raw_config, resolve=True)
    )


def _prepared_batch(
    batch: Mapping[str, Any], *, config: MasterConfig, processor: Any
) -> BatchedDataDict[Any]:
    if "packed_message_log" in batch:
        raise ValueError(
            "The loader-topology benchmark requires Energon packing to be disabled."
        )
    return prepare_sft_batch(
        batch,
        tokenizer=processor.tokenizer,
        only_unmask_final=config.sft.only_unmask_final,
        make_sequence_length_divisible_by=config.policy[
            "make_sequence_length_divisible_by"
        ],
    )


def _batch_counts(batch: Mapping[str, Any]) -> tuple[int, int]:
    input_lengths = batch.get("input_lengths")
    if not isinstance(input_lengths, torch.Tensor) or input_lengths.ndim != 1:
        raise ValueError("Prepared SFT batch has no rank-1 input_lengths tensor.")
    return input_lengths.numel(), int(input_lengths.sum().item())


def _throughput_record(
    *,
    topology: str,
    timings: list[float],
    samples: int,
    tokens: int,
    build_seconds: float,
    num_loaders: int,
    workers_per_loader: int,
) -> dict[str, Any]:
    summary = _timing_summary(timings)
    measured_seconds = sum(timings)
    return {
        "topology": topology,
        "timing": summary,
        "build_seconds": build_seconds,
        "num_loaders": num_loaders,
        "workers_per_loader": workers_per_loader,
        "total_configured_workers": num_loaders * workers_per_loader,
        "samples": samples,
        "tokens": tokens,
        "samples_per_second": samples / measured_seconds,
        "tokens_per_second": tokens / measured_seconds,
    }


def _run_loader_ablation(
    *,
    config_path: str,
    overrides: list[str],
    dp_size: int,
    global_batch_size_override: int | None,
    global_loader_workers: int,
    rank_local_loader_workers: int,
    warmup: int,
    steps: int,
    records: list[dict[str, Any]],
) -> None:
    config = _load_master_config(config_path, overrides)
    if config.data["backend"] != "energon":
        raise ValueError("The loader benchmark requires data.backend=energon.")
    energon_config = config.data["energon"]
    packing = energon_config.task_encoder.packing
    if packing is not None:
        raise ValueError(
            "The loader-topology benchmark compares sample placement. Set "
            "data.energon.task_encoder.packing=null for this ablation."
        )
    global_batch_size = (
        global_batch_size_override
        if global_batch_size_override is not None
        else config.policy["train_global_batch_size"]
    )
    if global_batch_size % dp_size:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by DP size "
            f"{dp_size}."
        )
    max_sequence_length = config.data["max_input_seq_length"]
    if max_sequence_length is None:
        raise ValueError("Energon SFT requires data.max_input_seq_length.")
    processor = get_tokenizer(config.policy["tokenizer"], get_processor=True)
    if rank_local_loader_workers > 0 and warmup == 0:
        raise ValueError(
            "loader-warmup must be at least 1 when Energon uses multiprocessing "
            "workers. Rank-local workers must start sequentially before concurrent "
            "iteration."
        )
    local_batch_size = global_batch_size // dp_size
    global_data_config = config.data.copy()
    global_data_config["energon"] = energon_config.model_copy(
        update={"num_workers": global_loader_workers}
    )
    rank_local_data_config = config.data.copy()
    rank_local_data_config["energon"] = energon_config.model_copy(
        update={"num_workers": rank_local_loader_workers}
    )

    single_build_started = time.perf_counter()
    single_loader = build_energon_sft_loader(
        data_config=global_data_config,
        source=config.data["train"],
        processor=processor,
        batch_size=global_batch_size,
        max_sequence_length=max_sequence_length,
        split_role="train",
        logical_rank=0,
        logical_world_size=1,
        placement_fingerprint="ablate_energon_single_global_loader",
    )
    single_iterator = iter(single_loader)
    single_build_seconds = time.perf_counter() - single_build_started

    def consume_single() -> tuple[int, int]:
        prepared = _prepared_batch(
            next(single_iterator), config=config, processor=processor
        )
        sample_count, token_count = _batch_counts(prepared)
        shards = prepared.shard_by_batch_size(
            shards=dp_size,
            batch_size=global_batch_size,
        )
        if len(shards) != dp_size:
            raise RuntimeError("Global loader did not produce one shard per DP rank.")
        return sample_count, token_count

    for _ in range(warmup):
        consume_single()
    single_times: list[float] = []
    single_samples = 0
    single_tokens = 0
    for _ in range(steps):
        started = time.perf_counter()
        sample_count, token_count = consume_single()
        single_times.append(time.perf_counter() - started)
        single_samples += sample_count
        single_tokens += token_count

    del single_iterator, single_loader
    gc.collect()

    parallel_build_started = time.perf_counter()
    rank_loaders = [
        build_energon_sft_loader(
            data_config=rank_local_data_config,
            source=config.data["train"],
            processor=processor,
            batch_size=local_batch_size,
            max_sequence_length=max_sequence_length,
            split_role="train",
            logical_rank=rank,
            logical_world_size=dp_size,
            placement_fingerprint="ablate_energon_rank_local_loaders",
        )
        for rank in range(dp_size)
    ]
    rank_iterators = [iter(loader) for loader in rank_loaders]
    parallel_build_seconds = time.perf_counter() - parallel_build_started

    def consume_rank(rank: int) -> tuple[int, int]:
        prepared = _prepared_batch(
            next(rank_iterators[rank]), config=config, processor=processor
        )
        return _batch_counts(prepared)

    parallel_times: list[float] = []
    parallel_samples = 0
    parallel_tokens = 0
    remaining_warmup = warmup
    if rank_local_loader_workers > 0:
        for rank in range(dp_size):
            consume_rank(rank)
        remaining_warmup -= 1
    with ThreadPoolExecutor(max_workers=dp_size) as executor:
        for _ in range(remaining_warmup):
            list(executor.map(consume_rank, range(dp_size)))
        for _ in range(steps):
            started = time.perf_counter()
            counts = list(executor.map(consume_rank, range(dp_size)))
            parallel_times.append(time.perf_counter() - started)
            parallel_samples += sum(count[0] for count in counts)
            parallel_tokens += sum(count[1] for count in counts)

    single_record = _throughput_record(
        topology="one_global_loader_then_shard",
        timings=single_times,
        samples=single_samples,
        tokens=single_tokens,
        build_seconds=single_build_seconds,
        num_loaders=1,
        workers_per_loader=global_loader_workers,
    )
    parallel_record = _throughput_record(
        topology="eight_rank_local_loaders",
        timings=parallel_times,
        samples=parallel_samples,
        tokens=parallel_tokens,
        build_seconds=parallel_build_seconds,
        num_loaders=dp_size,
        workers_per_loader=rank_local_loader_workers,
    )
    _log_record(
        records,
        {
            "benchmark": "loader_topology",
            "configuration": {
                "config_path": config_path,
                "overrides": overrides,
                "dp_size": dp_size,
                "global_batch_size": global_batch_size,
                "local_batch_size": local_batch_size,
                "max_sequence_length": max_sequence_length,
                "global_loader_workers": global_loader_workers,
                "rank_local_loader_workers": rank_local_loader_workers,
                "warmup": warmup,
                "steps": steps,
                "packing": None,
            },
            "one_global_loader_then_shard": single_record,
            "rank_local_loaders": parallel_record,
            "rank_local_over_global_samples_per_second": parallel_record[
                "samples_per_second"
            ]
            / single_record["samples_per_second"],
            "rank_local_over_global_tokens_per_second": parallel_record[
                "tokens_per_second"
            ]
            / single_record["tokens_per_second"],
        },
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("packing", "loaders", "all"),
        default="packing",
    )
    parser.add_argument(
        "--config",
        help="SFT YAML required for --mode loaders or --mode all",
    )
    parser.add_argument("--dp-size", type=_positive_int, default=8)
    parser.add_argument(
        "--global-batch-sizes",
        type=_positive_int,
        nargs="+",
        default=[64, 128, 256],
    )
    parser.add_argument(
        "--max-sequence-lengths",
        type=_positive_int,
        nargs="+",
        default=[8_192, 32_768, 131_072],
    )
    parser.add_argument("--algorithm", default="first_fit_decreasing")
    parser.add_argument(
        "--sequence-length-pad-multiple", type=_positive_int, default=128
    )
    parser.add_argument("--packing-warmup", type=_nonnegative_int, default=3)
    parser.add_argument("--packing-repeats", type=_positive_int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loader-warmup", type=_nonnegative_int, default=2)
    parser.add_argument("--loader-steps", type=_positive_int, default=50)
    parser.add_argument(
        "--loader-global-batch-size",
        type=_positive_int,
        help="Override policy.train_global_batch_size for the loader ablation",
    )
    parser.add_argument(
        "--global-loader-workers",
        type=_nonnegative_int,
        default=8,
        help="Worker count for the single global loader",
    )
    parser.add_argument(
        "--rank-local-loader-workers",
        type=_nonnegative_int,
        default=1,
        help="Worker count for each rank-local loader",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON file for all benchmark records; stdout is always logged",
    )
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    if args.mode in {"loaders", "all"} and not args.config:
        parser.error("--config is required for loader ablations")
    if args.mode in {"loaders", "all"} and args.loader_steps < 50:
        parser.error("--loader-steps must be at least 50 for loader ablations")
    return args, overrides


def main() -> None:
    args, overrides = _parse_args()
    records: list[dict[str, Any]] = []
    _log_record(
        records,
        {
            "benchmark": "environment",
            "configuration": {
                "mode": args.mode,
                "dp_size": args.dp_size,
                "torch_version": torch.__version__,
                "cpu_count": os.cpu_count(),
            },
        },
    )
    if args.mode in {"packing", "all"}:
        _run_packing_ablation(
            dp_size=args.dp_size,
            global_batch_sizes=args.global_batch_sizes,
            max_sequence_lengths=args.max_sequence_lengths,
            algorithm=args.algorithm,
            sequence_length_pad_multiple=args.sequence_length_pad_multiple,
            warmup=args.packing_warmup,
            repeats=args.packing_repeats,
            seed=args.seed,
            records=records,
        )
    if args.mode in {"loaders", "all"}:
        assert args.config is not None
        _run_loader_ablation(
            config_path=args.config,
            overrides=overrides,
            dp_size=args.dp_size,
            global_batch_size_override=args.loader_global_batch_size,
            global_loader_workers=args.global_loader_workers,
            rank_local_loader_workers=args.rank_local_loader_workers,
            warmup=args.loader_warmup,
            steps=args.loader_steps,
            records=records,
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
