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

"""Modified first-fit-decreasing sequence packing."""

from bisect import bisect

from nemo_rl.data.packing.base import SequencePacker

IndexedCost = tuple[int, int]


class ModifiedFirstFitDecreasingPacker(SequencePacker):
    """Apply the Johnson and Garey modified first-fit-decreasing heuristic."""

    def _classify_items(
        self, items: list[IndexedCost]
    ) -> tuple[
        list[IndexedCost],
        list[IndexedCost],
        list[IndexedCost],
        list[IndexedCost],
    ]:
        large: list[IndexedCost] = []
        medium: list[IndexedCost] = []
        small: list[IndexedCost] = []
        tiny: list[IndexedCost] = []
        for source_index, size in items:
            if size > self.bin_capacity / 2:
                large.append((source_index, size))
            elif size > self.bin_capacity / 3:
                medium.append((source_index, size))
            elif size > self.bin_capacity / 6:
                small.append((source_index, size))
            else:
                tiny.append((source_index, size))
        return large, medium, small, tiny

    def _pack_implementation(self, sequence_lengths: list[int]) -> list[list[int]]:
        items = [
            (source_index, length)
            for source_index, length in enumerate(sequence_lengths)
        ]
        large, medium, small, tiny = self._classify_items(items)

        # Length-only sorts keep original source order when costs are equal.
        large.sort(key=lambda item: -item[1])
        medium.sort(key=lambda item: -item[1])
        small.sort(key=lambda item: item[1])
        tiny.sort(key=lambda item: item[1])

        bins: list[list[IndexedCost]] = [[item] for item in large]

        for bin_contents in bins:
            remaining = self.bin_capacity - sum(size for _, size in bin_contents)
            for item_index, (_, size) in enumerate(medium):
                if size <= remaining:
                    bin_contents.append(medium.pop(item_index))
                    break

        for bin_contents in reversed(bins):
            has_medium = any(
                self.bin_capacity / 3 < size <= self.bin_capacity / 2
                for _, size in bin_contents
            )
            if has_medium or len(small) < 2:
                continue
            remaining = self.bin_capacity - sum(size for _, size in bin_contents)
            if small[0][1] + small[1][1] > remaining:
                continue
            first_small = small.pop(0)
            second_index = next(
                (
                    index
                    for index in range(len(small) - 1, -1, -1)
                    if small[index][1] <= remaining - first_small[1]
                ),
                None,
            )
            if second_index is not None:
                bin_contents.extend([first_small, small.pop(second_index)])

        remaining_items = sorted(medium + small + tiny, key=lambda item: -item[1])
        for bin_contents in bins:
            while remaining_items:
                remaining = self.bin_capacity - sum(size for _, size in bin_contents)
                if remaining < remaining_items[-1][1]:
                    break
                chosen_index = next(
                    (
                        index
                        for index, (_, size) in enumerate(remaining_items)
                        if size <= remaining
                    ),
                    None,
                )
                if chosen_index is None:
                    break
                bin_contents.append(remaining_items.pop(chosen_index))

        ffd_bins: list[list[IndexedCost]] = [[]]
        ffd_bin_sizes = [0]
        for source_index, size in sorted(remaining_items, key=lambda item: -item[1]):
            if size <= self.bin_capacity - ffd_bin_sizes[0]:
                new_bin = ffd_bins.pop(0)
                new_bin_size = ffd_bin_sizes.pop(0)
            else:
                new_bin = []
                new_bin_size = 0
            new_bin.append((source_index, size))
            new_bin_size += size
            insert_index = bisect(ffd_bin_sizes, new_bin_size)
            ffd_bins.insert(insert_index, new_bin)
            ffd_bin_sizes.insert(insert_index, new_bin_size)

        bins.extend(ffd_bins)
        return [
            [source_index for source_index, _ in bin_contents]
            for bin_contents in bins
            if bin_contents
        ]


__all__ = ["ModifiedFirstFitDecreasingPacker"]
