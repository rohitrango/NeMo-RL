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

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from nemo_rl.data.energon.config import EnergonPackingOptions
from nemo_rl.data.energon.multimodal.packing.base import EnergonPackingHooks
from nemo_rl.data.energon.multimodal.types import EncodedSFTSample, PackedSFTSample

_SAMPLE_SCHEMA = "nemo_rl.sft.encoded.v1"


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def first_fit_multimodal(
    samples: Sequence[EncodedSFTSample],
    *,
    max_sequence_length: int,
    sequence_length_pad_multiple: int,
) -> list[list[EncodedSFTSample]]:
    """Pack compatible samples by descending aligned length, deterministically."""
    if max_sequence_length <= 0 or sequence_length_pad_multiple <= 0:
        raise ValueError("Packing length and alignment must be positive.")

    indexed_samples: list[tuple[int, int, EncodedSFTSample]] = []
    for index, sample in enumerate(samples):
        padded_length = _round_up(sample.length, sequence_length_pad_multiple)
        if padded_length > max_sequence_length:
            raise ValueError(
                f"Sample {sample.sample_key!r} aligned length {padded_length} "
                f"exceeds pack capacity {max_sequence_length}."
            )
        indexed_samples.append((index, padded_length, sample))

    # Python's sort is stable, so equal-size candidates retain their source order.
    indexed_samples.sort(key=lambda item: -item[1])
    packs: list[list[EncodedSFTSample]] = []
    pack_lengths: list[int] = []
    pack_groups: list[tuple[object, ...]] = []
    for _, padded_length, sample in indexed_samples:
        for pack_index, (used, group_key) in enumerate(
            zip(pack_lengths, pack_groups)
        ):
            if (
                group_key == sample.group_key
                and used + padded_length <= max_sequence_length
            ):
                packs[pack_index].append(sample)
                pack_lengths[pack_index] += padded_length
                break
        else:
            packs.append([sample])
            pack_lengths.append(padded_length)
            pack_groups.append(sample.group_key)

    if sum(map(len, packs)) != len(samples):
        raise RuntimeError("Energon first-fit packing did not preserve all samples.")
    return packs


def pack_selected_multimodal(
    samples: list[EncodedSFTSample],
    *,
    max_sequence_length: int,
    sequence_length_pad_multiple: int,
) -> PackedSFTSample:
    """Construct one restore-aware physical pack from a selected group."""
    if not samples:
        raise ValueError("Cannot construct an empty Energon SFT pack.")
    group_key = samples[0].group_key
    if any(sample.group_key != group_key for sample in samples[1:]):
        raise ValueError("One Energon SFT pack cannot mix compatibility groups.")
    padded_lengths = [
        _round_up(sample.length, sequence_length_pad_multiple) for sample in samples
    ]
    if sum(padded_lengths) > max_sequence_length:
        raise ValueError(
            "Selected Energon SFT samples exceed the configured pack capacity."
        )
    return PackedSFTSample.derive_from(
        samples[0],
        __key__=",".join(sample.sample_key for sample in samples),
        samples=list(samples),
        source_lengths=[sample.length for sample in samples],
        source_padded_lengths=padded_lengths,
        source_ids=[sample.sample_key for sample in samples],
        group_key=group_key,
        pack_capacity=max_sequence_length,
    )


def build_packing_hooks(
    options: EnergonPackingOptions,
) -> EnergonPackingHooks[EncodedSFTSample, EncodedSFTSample, PackedSFTSample]:
    """Build the registered first-fit callbacks for one loader instance."""

    return EnergonPackingHooks(
        key="first_fit_multimodal",
        version="1",
        sample_schema=_SAMPLE_SCHEMA,
        select_samples_to_pack=partial(
            first_fit_multimodal,
            max_sequence_length=options.max_sequence_length,
            sequence_length_pad_multiple=options.sequence_length_pad_multiple,
        ),
        pack_selected_samples=partial(
            pack_selected_multimodal,
            max_sequence_length=options.max_sequence_length,
            sequence_length_pad_multiple=options.sequence_length_pad_multiple,
        ),
    )


__all__ = [
    "build_packing_hooks",
    "first_fit_multimodal",
    "pack_selected_multimodal",
]
