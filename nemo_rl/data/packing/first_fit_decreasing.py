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

"""First-fit-decreasing sequence packing."""

from nemo_rl.data.packing.base import FirstFitPacker


class FirstFitDecreasingPacker(FirstFitPacker):
    """Place descending-cost items into the first bin where each item fits."""

    def _prepare_sequences(self, sequence_lengths: list[int]) -> list[tuple[int, int]]:
        indexed_lengths = [
            (length, source_index)
            for source_index, length in enumerate(sequence_lengths)
        ]
        indexed_lengths.sort(key=lambda item: -item[0])
        return indexed_lengths


__all__ = ["FirstFitDecreasingPacker"]
