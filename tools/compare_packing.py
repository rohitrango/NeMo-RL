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

"""Compare greedy-knapsack packing across the two stacks at identical settings.

tools/compare_task_encoders.py stops before packing. This takes the same rows
one step further and answers two separate questions:

  1. do the two stacks assign the same COST to each sample?
  2. given identical costs and capacity, do the two greedy-knapsack
     implementations produce the same BINS?

Keeping them apart matters: a membership difference caused by disagreeing costs
is a different bug from one caused by the algorithm.

Cost inputs, and why they need care:

  reference  select_samples_to_pack uses sample.total_len_padded, which is
             total_len plus context-parallel / FP8 padding. With CP=1 and
             sequence_parallel off it equals total_len.
  nemo_rl    packing/sft.py _aligned_packing_cost rounds packing_cost up to
             sequence_length_pad_multiple (8 in the recipe).

So by default NeMo-RL pads every sample and the reference does not. Pass
--pad-multiple 1 to strip that and isolate the algorithm; pass 8 to see the
effect of the recipe's setting.

Both implementations are structurally the same greedy descent, but they break
ties differently, which is the thing this script is really looking for:

  reference  knapsacks.py greedy_knapsack -- sorts by size only; a stable sort
             plus rightmost-fit pops the LAST of an equal-cost group
  nemo_rl    greedy_knapsack.py -- sorts by (length, -source_index), so the
             rightmost fit is the SMALLEST source index of that group

Usage:
    python tools/compare_packing.py --limit 20 --pad-multiple 1
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import compare_cookers as cc  # noqa: E402
import compare_task_encoders as harness  # noqa: E402


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def collect_costs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Cook and pre-encode rows, returning one cost record per sample."""
    import importlib

    reference_root = Path(args.reference_root).expanduser().resolve()
    sys.path.insert(0, str(reference_root))
    sys.path.insert(0, str(reference_root / "examples" / "multimodal"))

    from megatron.energon import WorkerConfig

    harness.install_megatron_core_stubs(reference_root, verbose=False)

    WorkerConfig.active_worker_config = None
    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
    worker_config.worker_activate(0)

    shared = Path(args.shared_tokenizer).expanduser() if args.shared_tokenizer else None
    nrl_encoder = harness.build_nemo_rl_encoder(
        Path(args.config).expanduser(),
        Path(args.model_path).expanduser(),
        shared,
        {"tiling_augment_prob": 0.0},
    )
    ref_encoder, _ = harness.build_reference_encoder(
        Path(args.reference_args).expanduser(),
        harness.build_reference_tokenizer(
            reference_root, shared or Path(args.model_path).expanduser()
        ),
    )

    ref_modules = [importlib.import_module(m) for m in harness.REFERENCE_MODULES]
    nrl_modules = [importlib.import_module(m) for m in cc.NEMO_RL_MODULES]

    def lookup(modules, name):
        for module in modules:
            found = getattr(module, name, None)
            if found is not None:
                return found
        return None

    records: list[dict[str, Any]] = []
    leaves = cc.load_leaves(Path(args.subset).expanduser(), args.split)
    if args.leaf:
        leaves = [leaf for leaf in leaves if args.leaf in leaf["name"]]

    for leaf in leaves:
        pair = cc.COOK_PAIRS.get(leaf["cook"])
        if pair is None:
            continue
        ref_cook = lookup(ref_modules, pair[0])
        nrl_cook = lookup(nrl_modules, pair[1])
        if ref_cook is None or nrl_cook is None:
            continue
        kwargs: dict[str, Any] = {}
        if "media_source" in pair[2]:
            kwargs["media_source"] = (
                None if leaf["media_root"] is None else cc.StubStore(leaf["media_root"])
            )
        if "aux" in pair[2]:
            kwargs.update(
                {k: cc.StubStore(v) for k, v in leaf["aux"].items()}
            )
        if "primary" in pair[2]:
            kwargs["primary"] = cc.StubStore(leaf["jsonl"].parent)

        for index, payload in cc.iter_rows(leaf["jsonl"], args.limit):
            if index is None:
                break
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    ref_cooked = ref_cook(
                        cc.build_sample(leaf, index, copy.deepcopy(payload)),
                        cache=cc.StubCache(),
                        **kwargs,
                    )
                    nrl_cooked = nrl_cook(
                        cc.build_sample(leaf, index, copy.deepcopy(payload)),
                        cache=cc.StubCache(),
                        **kwargs,
                    )
                except Exception:  # noqa: BLE001 - rows both stacks reject
                    continue
                worker_config.worker_push_sample_index(index)
                try:
                    ref_encoded = ref_encoder.preencode_sample(ref_cooked)
                    nrl_encoded = nrl_encoder.preencode_sample(nrl_cooked)
                except Exception:  # noqa: BLE001
                    continue
                finally:
                    worker_config.worker_pop_sample_index()

            records.append(
                {
                    "key": f"{leaf['name']}/{index:06d}",
                    # What select_samples_to_pack actually consumes on each side.
                    "reference_cost": int(ref_encoded.total_len_padded),
                    "reference_unpadded": int(ref_encoded.total_len),
                    "nemo_rl_cost": _round_up(
                        int(nrl_encoded.packing_cost), args.pad_multiple
                    ),
                    "nemo_rl_unpadded": int(nrl_encoded.packing_cost),
                }
            )
    return records


def run(args: argparse.Namespace) -> int:
    logging.disable(logging.WARNING)
    records = collect_costs(args)
    if not records:
        print("no samples collected")
        return 2

    print(f"samples: {len(records)}   pad_multiple: {args.pad_multiple}")

    # --- question 1: do the costs agree? -----------------------------------
    cost_mismatch = [r for r in records if r["reference_cost"] != r["nemo_rl_cost"]]
    print("\n--- cost agreement ---")
    print(f"  matching : {len(records) - len(cost_mismatch)}/{len(records)}")
    if cost_mismatch:
        print(f"  differing: {len(cost_mismatch)}")
        for record in cost_mismatch[:5]:
            print(
                f"    {record['key']}: reference {record['reference_cost']} "
                f"(unpadded {record['reference_unpadded']}) vs nemo_rl "
                f"{record['nemo_rl_cost']} (unpadded {record['nemo_rl_unpadded']})"
            )

    # --- question 2: same costs in, same bins out? -------------------------
    # Feed BOTH algorithms the reference costs, so any difference here is the
    # algorithm and not the cost.
    costs = [r["reference_cost"] for r in records]
    keys = [r["key"] for r in records]

    import data_loading.knapsacks as reference_knapsacks

    from nemo_rl.data.packing.factory import get_packer

    if args.algorithm == "balanced_greedy_knapsack":
        ref_bins = reference_knapsacks.balanced_greedy_knapsack(
            list(costs), list(keys), args.capacity, args.delta
        )
        packer = get_packer(
            args.algorithm,
            args.capacity,
            balanced_knapsack_delta=args.delta,
        )
    else:
        ref_bins = reference_knapsacks.greedy_knapsack(
            list(costs), list(keys), args.capacity
        )
        packer = get_packer(args.algorithm, args.capacity)

    nrl_bins_idx = packer.pack(list(costs))
    nrl_bins = [[keys[i] for i in b] for b in nrl_bins_idx]

    duplicates = Counter(costs)
    tied = sum(count for count in duplicates.values() if count > 1)
    oversized = [c for c in costs if c > args.capacity]

    print("\n--- packing with identical costs and capacity ---")
    print(f"  algorithm    : {args.algorithm}"
          + (f"  delta={args.delta}" if args.algorithm.startswith("balanced") else ""))
    print(f"  capacity     : {args.capacity}")
    print(
        f"  tied costs   : {tied}/{len(costs)} samples share a cost with "
        f"another ({len(duplicates)} distinct costs)"
    )
    if oversized:
        # The reference raises on these; NeMo-RL may not. Report before packing
        # so an exception is not mistaken for a parity finding.
        print(
            f"  OVERSIZED    : {len(oversized)} samples exceed capacity "
            f"(largest {max(oversized)}); the reference raises on these"
        )
        return 2
    print(f"  bins         : reference {len(ref_bins)}   nemo_rl {len(nrl_bins)}")

    # balanced_greedy_knapsack pre-allocates ceil(total/cap)+delta bins, so
    # empty bins are expected on both sides; compare them and the non-empty
    # membership separately.
    ref_sets = sorted(tuple(sorted(b)) for b in ref_bins)
    nrl_sets = sorted(tuple(sorted(b)) for b in nrl_bins)
    ref_nonempty = [b for b in ref_sets if b]
    nrl_nonempty = [b for b in nrl_sets if b]
    print(f"  empty bins   : reference {len(ref_sets) - len(ref_nonempty)}   "
          f"nemo_rl {len(nrl_sets) - len(nrl_nonempty)}")
    print(f"  non-empty    : reference {len(ref_nonempty)}   nemo_rl {len(nrl_nonempty)}")
    print(f"  membership   : {'IDENTICAL' if ref_sets == nrl_sets else 'DIFFERS'}")

    ref_fill = sorted(sum(costs[keys.index(k)] for k in b) for b in ref_bins)
    nrl_fill = sorted(sum(costs[keys.index(k)] for k in b) for b in nrl_bins)
    print(f"  bin fills    : {'IDENTICAL' if ref_fill == nrl_fill else 'DIFFERS'}")

    if ref_sets != nrl_sets:
        only_ref = [b for b in ref_sets if b not in nrl_sets]
        only_nrl = [b for b in nrl_sets if b not in ref_sets]
        print(f"  bins only in reference: {len(only_ref)}")
        for b in only_ref[:3]:
            print(f"    {list(b)[:4]}{' ...' if len(b) > 4 else ''}")
        print(f"  bins only in nemo_rl  : {len(only_nrl)}")
        for b in only_nrl[:3]:
            print(f"    {list(b)[:4]}{' ...' if len(b) > 4 else ''}")
    compare_pack_construction(records, args)
    return 0 if (ref_sets == nrl_sets and not cost_mismatch) else 1


def compare_pack_construction(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    """Compare what each stack builds from one selected group of samples.

    The two carry the same information in different shapes:

      reference  cu_lengths / cu_lengths_padded -- running offsets, leading 0
      nemo_rl    source_lengths / source_padded_lengths -- per-sample values

    So the comparable quantity is the cumulative sum. The reference builds this
    in pack_selected_samples (task_encoder.py:1418); NeMo-RL in
    packing/sft.py:88.
    """
    from itertools import accumulate

    print("\n--- pack construction (cu_seqlens) ---")

    # One pack's worth of samples that fits the capacity, taken in order.
    chosen: list[dict[str, Any]] = []
    running = 0
    for record in records:
        if running + record["reference_cost"] > args.capacity:
            break
        chosen.append(record)
        running += record["reference_cost"]
    if not chosen:
        print("  no samples fit the capacity; nothing to construct")
        return

    unpadded = [r["reference_unpadded"] for r in chosen]
    ref_padded = [r["reference_cost"] for r in chosen]
    nrl_padded = [
        _round_up(r["nemo_rl_unpadded"], args.pad_multiple) for r in chosen
    ]

    ref_cu = [0] + list(accumulate(ref_padded))
    nrl_cu = [0] + list(accumulate(nrl_padded))

    print(f"  samples in pack      : {len(chosen)}")
    print(f"  pad multiple         : {args.pad_multiple}")
    print(f"  per-sample padded    : {'IDENTICAL' if ref_padded == nrl_padded else 'DIFFERS'}")
    print(f"  cu_seqlens           : {'IDENTICAL' if ref_cu == nrl_cu else 'DIFFERS'}")
    print(f"  pack total           : reference {ref_cu[-1]}   nemo_rl {nrl_cu[-1]}")
    print(f"  within capacity      : reference {ref_cu[-1] <= args.capacity}   "
          f"nemo_rl {nrl_cu[-1] <= args.capacity}")
    if ref_padded != nrl_padded:
        for index, (a, b, raw) in enumerate(zip(ref_padded, nrl_padded, unpadded)):
            if a != b:
                print(f"    first diff at {index}: unpadded {raw} -> "
                      f"reference {a}, nemo_rl {b}")
                break
    # Divisibility is what megatron asserts when it slices across CP ranks.
    for multiple in (8, 16):
        ok_ref = all(v % multiple == 0 for v in ref_padded)
        ok_nrl = all(v % multiple == 0 for v in nrl_padded)
        print(f"  all lengths %% {multiple:>2} == 0 : reference {ok_ref}   nemo_rl {ok_nrl}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", default="~/data/super-test-blend/cook_subset_v2/subset.yaml"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--leaf", default=None)
    parser.add_argument("--limit", type=int, default=20, help="Rows per leaf.")
    parser.add_argument("--reference-root", default="energon-megatron-lm")
    parser.add_argument("--reference-args", default="/tmp/ref_args.json")
    parser.add_argument("--shared-tokenizer", default="/tmp/shared_tokenizer")
    parser.add_argument(
        "--model-path",
        default="~/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    )
    parser.add_argument(
        "--config",
        default=(
            "examples/configs/sft_v2_tests/"
            "vlm_sft-nemotron-omni-30ba3b-4n8g-megatron-tp4etp4-super-test-blend.v1.yaml"
        ),
    )
    parser.add_argument(
        "--algorithm",
        default="greedy_knapsack",
        choices=["greedy_knapsack", "balanced_greedy_knapsack"],
        help="Run BOTH stacks with this algorithm.",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=5,
        help="balanced_knapsack_delta; the launch script passes 5.",
    )
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument(
        "--pad-multiple",
        type=int,
        default=1,
        help=(
            "NeMo-RL rounds each cost up to this. 1 isolates the algorithm; 8 "
            "reproduces the recipe's sequence_length_pad_multiple."
        ),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
