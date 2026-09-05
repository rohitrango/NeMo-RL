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

"""Run an exact-data SFT V1/V2 loss-parity diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

V1_CONFIG = "examples/configs/sft_v2_tests/sft_vlm_3B_energon.yaml"
V2_CONFIG = "examples/configs/sft_v2_tests/sft_vlm_3B_energon_v2.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _common_overrides(args: argparse.Namespace) -> list[str]:
    return [
        f"policy.model_name={args.model}",
        f"policy.train_global_batch_size={args.global_batch_size}",
        f"policy.train_micro_batch_size={args.micro_batch_size}",
        "policy.megatron_cfg.tensor_model_parallel_size=1",
        "policy.megatron_cfg.pipeline_model_parallel_size=1",
        "policy.megatron_cfg.context_parallel_size=1",
        f"policy.megatron_cfg.optimizer.lr={args.learning_rate}",
        f"policy.megatron_cfg.optimizer.min_lr={args.learning_rate}",
        f"policy.megatron_cfg.scheduler.lr_warmup_init={args.learning_rate}",
        "policy.megatron_cfg.scheduler.lr_warmup_iters=0",
        "policy.megatron_cfg.scheduler.lr_decay_style=constant",
        "sft.seed=42",
        "sft.only_unmask_final=false",
        f"sft.max_num_steps={args.steps}",
        "data.shuffle=false",
        f"data.energon.num_workers={args.loader_workers}",
        "checkpointing.enabled=false",
        "logger.tensorboard_enabled=true",
        "logger.monitor_gpus=false",
        f"cluster.gpus_per_node={args.dp_size}",
        "cluster.num_nodes=1",
    ]


def _load_resolved(config_path: str, overrides: list[str]) -> dict[str, Any]:
    config = load_config(config_path)
    config = parse_hydra_overrides(config, overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"Resolved config {config_path!r} is not a mapping.")
    return resolved


def _parallelism(config: dict[str, Any]) -> dict[str, int]:
    policy = config["policy"]
    megatron = policy["megatron_cfg"]
    world_size = config["cluster"]["gpus_per_node"] * config["cluster"]["num_nodes"]
    model_parallel_size = (
        megatron["tensor_model_parallel_size"]
        * megatron["pipeline_model_parallel_size"]
        * megatron["context_parallel_size"]
    )
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"World size {world_size} is not divisible by model parallel size "
            f"{model_parallel_size}."
        )
    data_parallel_size = world_size // model_parallel_size
    global_batch_size = policy["train_global_batch_size"]
    micro_batch_size = policy["train_micro_batch_size"]
    denominator = data_parallel_size * micro_batch_size
    if global_batch_size % denominator != 0:
        raise ValueError(
            f"Global batch {global_batch_size} is not divisible by DP x microbatch "
            f"({data_parallel_size} x {micro_batch_size})."
        )
    return {
        "world_size": world_size,
        "model_parallel_size": model_parallel_size,
        "data_parallel_size": data_parallel_size,
        "gradient_accumulation_steps": global_batch_size // denominator,
    }


def _training_audit(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    compared = {
        "model_name": (v1["policy"]["model_name"], v2["policy"]["model_name"]),
        "tokenizer": (v1["policy"]["tokenizer"], v2["policy"]["tokenizer"]),
        "global_batch_size": (
            v1["policy"]["train_global_batch_size"],
            v2["policy"]["train_global_batch_size"],
        ),
        "micro_batch_size": (
            v1["policy"]["train_micro_batch_size"],
            v2["policy"]["train_micro_batch_size"],
        ),
        "max_sequence_length": (
            v1["policy"]["max_total_sequence_length"],
            v2["policy"]["max_total_sequence_length"],
        ),
        "seed": (v1["sft"]["seed"], v2["sft"]["seed"]),
        "only_unmask_final": (
            v1["sft"]["only_unmask_final"],
            v2["sft"]["only_unmask_final"],
        ),
        "optimizer": (
            v1["policy"]["megatron_cfg"]["optimizer"],
            v2["policy"]["megatron_cfg"]["optimizer"],
        ),
        "scheduler": (
            v1["policy"]["megatron_cfg"]["scheduler"],
            v2["policy"]["megatron_cfg"]["scheduler"],
        ),
        "parallelism": (_parallelism(v1), _parallelism(v2)),
        "data_preprocessing": (
            {
                "train": v1["data"]["train"],
                "max_input_seq_length": v1["data"]["max_input_seq_length"],
                "add_bos": v1["data"]["add_bos"],
                "add_eos": v1["data"]["add_eos"],
                "add_generation_prompt": v1["data"]["add_generation_prompt"],
                "shuffle": v1["data"]["shuffle"],
                "num_workers": v1["data"]["energon"]["num_workers"],
                "processor_adapter": v1["data"]["energon"]["processor_adapter"],
                "seed_offset": v1["data"]["energon"].get("seed_offset", 0),
            },
            {
                "train": v2["data"]["train"],
                "max_input_seq_length": v2["data"]["max_input_seq_length"],
                "add_bos": v2["data"]["add_bos"],
                "add_eos": v2["data"]["add_eos"],
                "add_generation_prompt": v2["data"]["add_generation_prompt"],
                "shuffle": v2["data"]["shuffle"],
                "num_workers": v2["data"]["energon"]["num_workers"],
                "processor_adapter": v2["data"]["energon"]["processor_adapter"],
                "seed_offset": v2["data"]["energon"].get("seed_offset", 0),
            },
        ),
    }
    mismatches = {
        name: {"v1": values[0], "v2": values[1]}
        for name, values in compared.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise AssertionError(
            "V1/V2 training settings differ:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )

    parallelism = compared["parallelism"][0]
    if parallelism["data_parallel_size"] <= 1:
        raise ValueError(
            "The DP diagnostic requires more than one data-parallel replica."
        )
    if compared["data_preprocessing"][0]["shuffle"]:
        raise ValueError("The DP diagnostic requires data.shuffle=false.")
    optimizer = compared["optimizer"][0]
    scheduler = compared["scheduler"][0]
    global_batch_size = compared["global_batch_size"][0]
    learning_rate = optimizer["lr"]
    if not (
        optimizer["min_lr"] == learning_rate
        and scheduler["lr_warmup_init"] == learning_rate
        and scheduler["lr_warmup_iters"] == 0
        and scheduler["lr_decay_style"] == "constant"
    ):
        raise AssertionError("The A/B requires one constant learning rate.")

    return {
        "model_name": compared["model_name"][0],
        "global_batch_size": global_batch_size,
        "micro_batch_size": compared["micro_batch_size"][0],
        "learning_rate": learning_rate,
        "learning_rate_over_global_batch": learning_rate / global_batch_size,
        "scheduler_increment_samples_per_step": global_batch_size,
        **parallelism,
    }


def _run(command: list[str], *, extra_env: dict[str, str] | None = None) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    env = os.environ.copy()
    env.pop("UV_NO_SYNC", None)
    if extra_env:
        env.update(extra_env)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def _energon_python() -> list[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("The A/B runner requires uv on PATH.")
    return [
        uv,
        "run",
        "--locked",
        "--extra",
        "energon",
        "--directory",
        str(REPO_ROOT),
        "python",
    ]


def _event_accumulator(log_dir: Path, required_tag: str) -> EventAccumulator:
    candidates = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard events found under {log_dir}.")
    loaded: list[tuple[int, EventAccumulator]] = []
    for path in candidates:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags()["scalars"]
        loaded.append((len(tags), accumulator))
        if required_tag in tags:
            return accumulator
    available = sorted(
        {tag for _, accumulator in loaded for tag in accumulator.Tags()["scalars"]}
    )
    raise KeyError(
        f"TensorBoard tag {required_tag!r} was not found under {log_dir}; "
        f"available tags: {available}."
    )


def _scalar_map(accumulator: EventAccumulator, tag: str) -> dict[int, float]:
    return {event.step: float(event.value) for event in accumulator.Scalars(tag)}


def _relative_delta(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-12)


def _compare_events(
    *,
    v1_log_dir: Path,
    v2_log_dir: Path,
    report_path: Path,
    loss_rtol: float,
    grad_rtol: float,
) -> dict[str, Any]:
    v1_events = _event_accumulator(v1_log_dir, "train/loss")
    v2_events = _event_accumulator(v2_log_dir, "loss")
    tag_pairs = {
        "loss": ("train/loss", "loss"),
        "grad_norm": ("train/grad_norm", "grad_norm"),
        "learning_rate": ("train/lr", "lr"),
        "valid_tokens": ("train/global_valid_toks", "global_valid_toks"),
        "valid_samples": ("train/global_valid_seqs", "global_valid_seqs"),
    }
    series: dict[str, Any] = {}
    failures: list[str] = []
    for name, (v1_tag, v2_tag) in tag_pairs.items():
        v1_values = _scalar_map(v1_events, v1_tag)
        v2_values = _scalar_map(v2_events, v2_tag)
        if v1_values.keys() != v2_values.keys():
            failures.append(
                f"{name}: step sets differ: {sorted(v1_values)} vs {sorted(v2_values)}"
            )
            common_steps = sorted(v1_values.keys() & v2_values.keys())
        else:
            common_steps = sorted(v1_values)
        rows = [
            {
                "step": step,
                "v1": v1_values[step],
                "v2": v2_values[step],
                "relative_delta": _relative_delta(v1_values[step], v2_values[step]),
            }
            for step in common_steps
        ]
        max_delta = max((row["relative_delta"] for row in rows), default=math.inf)
        series[name] = {"max_relative_delta": max_delta, "steps": rows}
        if name == "loss" and max_delta > loss_rtol:
            failures.append(f"loss: max relative delta {max_delta:.6g} > {loss_rtol}")
        elif name == "grad_norm" and max_delta > grad_rtol:
            failures.append(
                f"grad_norm: max relative delta {max_delta:.6g} > {grad_rtol}"
            )
        elif name in {"valid_tokens", "valid_samples"} and any(
            row["v1"] != row["v2"] for row in rows
        ):
            failures.append(f"{name}: V1 and V2 counts differ")
        elif name == "learning_rate" and any(
            not math.isclose(row["v1"], row["v2"], rel_tol=0.0, abs_tol=1.0e-12)
            for row in rows
        ):
            failures.append("learning_rate: V1 and V2 values differ")

    report = {
        "passed": not failures,
        "failures": failures,
        "v1_log_dir": str(v1_log_dir),
        "v2_log_dir": str(v2_log_dir),
        "series": series,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "failures": failures}, indent=2))
    print(f"A/B report: {report_path}")
    return report


def _logger_overrides(*, log_dir: Path) -> list[str]:
    return [f"logger.log_dir={log_dir}", "logger.wandb_enabled=false"]


def _run_ab(args: argparse.Namespace) -> None:
    if args.dp_size <= 1 or args.steps <= 0:
        raise ValueError("DP size must exceed one and steps must be positive.")
    if args.global_batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("Global and micro batch sizes must be positive.")
    if args.loader_workers < 0:
        raise ValueError("Loader workers must be non-negative.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    register_omegaconf_resolvers()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = _common_overrides(args)
    v1_overrides = [
        *common,
        "sft.val_at_start=false",
        "sft.val_at_end=false",
        "sft.val_period=0",
        *_logger_overrides(log_dir=output_dir / "v1"),
    ]
    v2_overrides = [
        *common,
        *_logger_overrides(log_dir=output_dir / "v2"),
    ]
    v1_config = _load_resolved(V1_CONFIG, v1_overrides)
    v2_config = _load_resolved(V2_CONFIG, v2_overrides)
    audit = _training_audit(v1_config, v2_config)
    (output_dir / "training_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print("Training settings match:")
    print(json.dumps(audit, indent=2, sort_keys=True))

    energon_python = _energon_python()
    capture_command = [
        *energon_python,
        "tools/sft_v2_reference.py",
        "compare-dp",
        "--config",
        V1_CONFIG,
        "--output",
        str(output_dir / "exact_dp_batches.json"),
        "--steps",
        str(args.steps),
        "--logical-world-size",
        str(audit["data_parallel_size"]),
    ]
    for override in common:
        capture_command.extend(["--override", override])
    _run(capture_command)

    _run(
        [
            *energon_python,
            "tools/run_sft_v1_dp_reference.py",
            "--config",
            V1_CONFIG,
            *v1_overrides,
        ],
        extra_env={"SFT_REFERENCE_DP_SIZE": str(audit["data_parallel_size"])},
    )
    _run(
        [
            *energon_python,
            "examples/run_sft_v2.py",
            "--config",
            V2_CONFIG,
            *v2_overrides,
        ]
    )
    report = _compare_events(
        v1_log_dir=output_dir / "v1",
        v2_log_dir=output_dir / "v2",
        report_path=output_dir / "comparison.json",
        loss_rtol=args.loss_rtol,
        grad_rtol=args.grad_rtol,
    )
    if not report["passed"]:
        raise SystemExit(1)


def _default_output_dir() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"results/sft_v1_v2_ab/{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run exact-data V1 and V2 training.")
    run.add_argument("--model", default="/data/models/Qwen2.5-VL-3B-Instruct")
    run.add_argument(
        "--dp-size",
        "--gpus",
        dest="dp_size",
        type=int,
        default=8,
        help="Number of GPUs and data-parallel replicas; TP/PP/CP remain 1.",
    )
    run.add_argument("--steps", type=int, default=10)
    run.add_argument("--global-batch-size", type=int, default=32)
    run.add_argument("--micro-batch-size", type=int, default=1)
    run.add_argument(
        "--loader-workers",
        type=int,
        default=1,
        help=(
            "Energon workers per loader; kept small because DP has one loader per GPU."
        ),
    )
    run.add_argument("--learning-rate", type=float, default=5.0e-6)
    run.add_argument("--output-dir", default=_default_output_dir())
    run.add_argument("--loss-rtol", type=float, default=1.0e-3)
    run.add_argument("--grad-rtol", type=float, default=5.0e-3)

    compare = subparsers.add_parser(
        "compare", help="Compare TensorBoard outputs from existing runs."
    )
    compare.add_argument("--v1-log-dir", type=Path, required=True)
    compare.add_argument("--v2-log-dir", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    compare.add_argument("--loss-rtol", type=float, default=1.0e-3)
    compare.add_argument("--grad-rtol", type=float, default=5.0e-3)
    args = parser.parse_args()
    if args.command == "run":
        _run_ab(args)
        return
    report = _compare_events(
        v1_log_dir=args.v1_log_dir,
        v2_log_dir=args.v2_log_dir,
        report_path=args.report,
        loss_rtol=args.loss_rtol,
        grad_rtol=args.grad_rtol,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
