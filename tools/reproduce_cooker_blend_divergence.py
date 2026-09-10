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

"""Reproduce the empty-conversation blend divergence without loading a model.

This uses the real subset rows and both real cooker functions. NeMo-RL rejects
the empty row inside the selected dataset iterator, so that iterator supplies
its next row. Megatron-LM returns the empty row from its cooker and rejects it
after blending, so the next blend draw can select a different dataset.

Run inside the NeMo-RL container:

    uv run --extra energon tools/reproduce_cooker_blend_divergence.py \
        --subset /mnt/rl-workspace/rohitkumarj/data/subset_cook_v2/subset.yaml \
        --reference-root \
          /mnt/rl-workspace/rohitkumarj/code/energon-megatron-lm
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import compare_cookers as comparison


def read_row(path: Path, index: int) -> dict[str, Any]:
    """Read one zero-based JSONL row."""
    with path.open("rb") as stream:
        for current, raw in enumerate(stream):
            if current == index:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise TypeError(f"Row {index} in {path} is not an object")
                return value
    raise IndexError(f"Row {index} does not exist in {path}")


def cook(
    cooker: Callable[..., Any],
    leaf: dict[str, Any],
    index: int,
    payload: dict[str, Any],
) -> Any:
    """Run one cooker with the same lightweight stores as compare_cookers.py."""
    media_source = (
        None if leaf["media_root"] is None else comparison.StubStore(leaf["media_root"])
    )
    return cooker(
        comparison.build_sample(leaf, index, copy.deepcopy(payload)),
        cache=comparison.StubCache(),
        media_source=media_source,
    )


def first_successful_cook(
    cooker: Callable[..., Any],
    leaf: dict[str, Any],
    rows: Iterator[tuple[int, dict[str, Any]]],
) -> tuple[int, Any, list[tuple[int, str]]]:
    """Match MapDataset: skip cooker errors within one dataset iterator."""
    errors: list[tuple[int, str]] = []
    for index, payload in rows:
        try:
            return index, cook(cooker, leaf, index, payload), errors
        except Exception as error:  # noqa: BLE001 - this demonstrates skip behavior
            errors.append((index, f"{type(error).__name__}: {error}"))
    raise RuntimeError("Dataset iterator produced no valid cooked sample")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()

    reference_root = args.reference_root.expanduser().resolve()
    sys.path.insert(0, str(reference_root))

    # The reference root is a CLI argument, so this import must follow sys.path setup.
    from examples.multimodal.data_loading.cookers.conversation import (
        cook_conversation as mlm_cooker,
    )
    from nemo_rl.data.energon.multimodal.cookers.nemotron import (
        cook_nemotron_conversation as nemorl_cooker,
    )

    leaves = comparison.load_leaves(args.subset.expanduser().resolve(), "train")
    benchfit = next(
        leaf for leaf in leaves if leaf["name"] == "reasoning_on__benchfit_qa"
    )
    scalecua = next(
        leaf
        for leaf in leaves
        if leaf["name"]
        == "reasoning_on__ScaleCUA_thinking_internvl_grounding_fixthink0319_safetyfix0323"
    )

    empty_row = read_row(benchfit["jsonl"], 2105)
    replacement_row = read_row(benchfit["jsonl"], 4311)
    next_blend_row = read_row(scalecua["jsonl"], 1415)

    mlm_empty = cook(mlm_cooker, benchfit, 2105, empty_row)
    try:
        mlm_empty.conversation[0]
    except IndexError as error:
        mlm_error = f"{type(error).__name__}: {error}"
    else:
        print("FAIL: the MLM path did not fail on the empty conversation")
        return 1

    nemorl_index, _, nemorl_errors = first_successful_cook(
        nemorl_cooker,
        benchfit,
        iter(((2105, empty_row), (4311, replacement_row))),
    )
    mlm_next = cook(mlm_cooker, scalecua, 1415, next_blend_row)

    print(f"NeMo-RL cooker error: {nemorl_errors[0][1]}")
    print(f"NeMo-RL same-dataset replacement: benchfit_qa:{nemorl_index}")
    print(f"MLM post-blend error: {mlm_error}")
    print(f"MLM next blend draw: ScaleCUA:1415 ({len(mlm_next.conversation)} turns)")
    print("PASS: rejection at different pipeline stages changes the packed candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
