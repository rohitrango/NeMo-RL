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

"""Concatenative sequence packing."""

from nemo_rl.data.packing.base import SequencePacker


class ConcatenativePacker(SequencePacker):
    """Append source-order items until the next item needs a new bin."""

    max_sequences_per_bin = -1

    def _pack_implementation(self, sequence_lengths: list[int]) -> list[list[int]]:
        bins: list[list[int]] = []
        current_bin: list[int] = []
        current_length = 0
        for source_index, length in enumerate(sequence_lengths):
            exceeds_capacity = current_length + length > self.bin_capacity
            exceeds_sequence_limit = (
                self.max_sequences_per_bin != -1
                and len(current_bin) >= self.max_sequences_per_bin
            )
            if exceeds_capacity or exceeds_sequence_limit:
                if current_bin:
                    bins.append(current_bin)
                current_bin = [source_index]
                current_length = length
            else:
                current_bin.append(source_index)
                current_length += length
        if current_bin:
            bins.append(current_bin)
        return bins


__all__ = ["ConcatenativePacker"]
