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

"""Model-neutral Energon adapter for shared SFT packing algorithms."""

from __future__ import annotations

from functools import partial

from nemo_rl.data.energon.config import EnergonPackingOptions
from nemo_rl.data.energon.multimodal.packing.base import EnergonPackingHooks
from nemo_rl.data.energon.multimodal.types import EncodedSFTSample, PackedSFTSample
from nemo_rl.data.packing import SequencePacker

_SAMPLE_SCHEMA = "nemo_rl.sft.encoded.v1"


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _aligned_packing_cost(
    sample: EncodedSFTSample, *, sequence_length_pad_multiple: int
) -> int:
    if sample.packing_cost < sample.length:
        raise ValueError(
            f"Sample {sample.sample_key!r} packing cost {sample.packing_cost} is "
            f"smaller than its token length {sample.length}."
        )
    return _round_up(sample.packing_cost, sequence_length_pad_multiple)


def select_samples_to_pack(
    samples: list[EncodedSFTSample],
    *,
    packer: SequencePacker,
    sequence_length_pad_multiple: int,
) -> list[list[EncodedSFTSample]]:
    """Select compatible physical packs with one shared packing algorithm."""
    if sequence_length_pad_multiple <= 0:
        raise ValueError("Packing alignment must be positive.")

    partitions: list[tuple[tuple[object, ...], list[EncodedSFTSample]]] = []
    for sample in samples:
        for group_key, group_samples in partitions:
            if sample.group_key == group_key:
                group_samples.append(sample)
                break
        else:
            partitions.append((sample.group_key, [sample]))

    packs: list[list[EncodedSFTSample]] = []
    selected_ids: list[int] = []
    for _, group_samples in partitions:
        costs = [
            _aligned_packing_cost(
                sample,
                sequence_length_pad_multiple=sequence_length_pad_multiple,
            )
            for sample in group_samples
        ]
        bins = packer.pack(costs)
        indexes = [index for bin_indexes in bins for index in bin_indexes]
        if sorted(indexes) != list(range(len(group_samples))):
            raise RuntimeError(
                "The shared packing algorithm did not preserve each source exactly once."
            )
        packs.extend(
            [[group_samples[index] for index in bin_indexes] for bin_indexes in bins]
        )
        selected_ids.extend(id(group_samples[index]) for index in indexes)

    if sorted(selected_ids) != sorted(id(sample) for sample in samples):
        raise RuntimeError("Energon SFT packing did not preserve all samples.")
    return packs


def pack_selected_samples(
    samples: list[EncodedSFTSample],
    *,
    max_sequence_length: int,
    sequence_length_pad_multiple: int,
) -> PackedSFTSample:
    """Build one restore-aware physical pack from a selected source group."""
    if not samples:
        raise ValueError("Cannot construct an empty Energon SFT pack.")
    group_key = samples[0].group_key
    if any(sample.group_key != group_key for sample in samples[1:]):
        raise ValueError("One Energon SFT pack cannot mix compatibility groups.")
    padded_lengths = [
        _aligned_packing_cost(
            sample,
            sequence_length_pad_multiple=sequence_length_pad_multiple,
        )
        for sample in samples
    ]
    if sum(padded_lengths) > max_sequence_length:
        raise ValueError(
            "Selected Energon SFT samples exceed the configured pack capacity."
        )
    source_ids = [sample.sample_key for sample in samples]
    return PackedSFTSample.derive_from(
        samples[0],
        __key__=",".join(source_ids),
        samples=list(samples),
        source_lengths=[sample.length for sample in samples],
        source_padded_lengths=padded_lengths,
        source_ids=source_ids,
        group_key=group_key,
        pack_capacity=max_sequence_length,
    )


def build_packing_hooks(
    options: EnergonPackingOptions,
    *,
    algorithm: str,
    version: str,
    packer_type: type[SequencePacker],
) -> EnergonPackingHooks[EncodedSFTSample, EncodedSFTSample, PackedSFTSample]:
    """Build Energon callbacks around one shared sequence packer."""
    packer = packer_type(options.max_sequence_length)
    return EnergonPackingHooks(
        key=algorithm,
        version=version,
        sample_schema=_SAMPLE_SCHEMA,
        select_samples_to_pack=partial(
            select_samples_to_pack,
            packer=packer,
            sequence_length_pad_multiple=options.sequence_length_pad_multiple,
        ),
        pack_selected_samples=partial(
            pack_selected_samples,
            max_sequence_length=options.max_sequence_length,
            sequence_length_pad_multiple=options.sequence_length_pad_multiple,
        ),
    )


__all__ = [
    "build_packing_hooks",
    "pack_selected_samples",
    "select_samples_to_pack",
]
