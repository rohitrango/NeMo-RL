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

"""Shared interfaces and helpers for sequence packing algorithms."""

from abc import ABC, abstractmethod

from nemo_rl.data.packing.metrics import PackingMetrics


class SequencePacker(ABC):
    """Arrange integer sequence costs into fixed-capacity bins."""

    def __init__(
        self,
        bin_capacity: int,
        collect_metrics: bool = False,
        min_bin_count: int | None = None,
        bin_count_multiple: int | None = None,
    ) -> None:
        """Initialize the sequence packer.

        Args:
            bin_capacity: Maximum cost of each bin.
            collect_metrics: Whether to collect metrics across packing calls.
            min_bin_count: Minimum number of non-empty bins.
            bin_count_multiple: Required multiple for the number of bins.

        Raises:
            ValueError: If a capacity or bin-count setting is invalid.
        """
        if bin_capacity <= 0:
            raise ValueError("bin_capacity must be positive")
        if min_bin_count is not None and min_bin_count < 0:
            raise ValueError("min_bin_count must be nonnegative")
        if bin_count_multiple is not None and bin_count_multiple < 1:
            raise ValueError("bin_count_multiple must be positive")

        self.bin_capacity = bin_capacity
        self.collect_metrics = collect_metrics
        self.min_bin_count = min_bin_count
        self.bin_count_multiple = bin_count_multiple
        self.metrics = PackingMetrics() if collect_metrics else None

    @abstractmethod
    def _pack_implementation(self, sequence_lengths: list[int]) -> list[list[int]]:
        """Return bins of indexes into ``sequence_lengths``."""

    def _adjust_bin_count(self, bins: list[list[int]]) -> list[list[int]]:
        """Increase the bin count to meet distributed packing constraints."""
        current_bin_count = len(bins)
        target_bin_count = current_bin_count
        if self.min_bin_count is not None:
            target_bin_count = max(target_bin_count, self.min_bin_count)
        if self.bin_count_multiple is not None:
            remainder = target_bin_count % self.bin_count_multiple
            if remainder:
                target_bin_count += self.bin_count_multiple - remainder
        if target_bin_count == current_bin_count:
            return bins

        total_sequences = sum(len(bin_contents) for bin_contents in bins)
        if total_sequences < target_bin_count:
            raise ValueError(
                f"Cannot create {target_bin_count} bins with only {total_sequences} "
                "sequences. Each bin must contain at least one sequence. Either reduce "
                "min_bin_count/bin_count_multiple or provide more sequences."
            )

        adjusted_bins = [bin_contents.copy() for bin_contents in bins]
        adjusted_bins.extend([] for _ in range(target_bin_count - current_bin_count))
        bin_sizes = sorted(
            (
                (len(bin_contents), index)
                for index, bin_contents in enumerate(adjusted_bins[:current_bin_count])
            ),
            reverse=True,
        )
        source_bin_index = 0
        for new_bin_index in range(current_bin_count, target_bin_count):
            while source_bin_index < len(bin_sizes):
                _, original_bin_index = bin_sizes[source_bin_index]
                if len(adjusted_bins[original_bin_index]) > 1:
                    adjusted_bins[new_bin_index].append(
                        adjusted_bins[original_bin_index].pop()
                    )
                    break
                source_bin_index += 1
            else:
                raise ValueError(
                    "Cannot create additional bins because sequences cannot be "
                    "redistributed. This indicates a packing bug."
                )
        return adjusted_bins

    def pack(self, sequence_lengths: list[int]) -> list[list[int]]:
        """Pack sequence costs and update metrics when enabled."""
        self._validate_sequence_lengths(sequence_lengths)
        bins = self._adjust_bin_count(self._pack_implementation(sequence_lengths))
        if self.metrics is not None:
            self.metrics.update(sequence_lengths, bins, self.bin_capacity)
        return bins

    def reset_metrics(self) -> None:
        """Reset collected metrics."""
        if self.metrics is not None:
            self.metrics.reset()

    def compute_metrics(
        self, sequence_lengths: list[int], bins: list[list[int]]
    ) -> dict[str, float]:
        """Calculate metrics for one packing result."""
        metrics = self.metrics or PackingMetrics()
        return metrics.calculate_stats_only(sequence_lengths, bins, self.bin_capacity)

    def get_aggregated_metrics(self) -> dict[str, float]:
        """Return metrics collected across packing calls."""
        return {} if self.metrics is None else self.metrics.get_aggregated_stats()

    def print_metrics(self) -> None:
        """Print collected metrics."""
        if self.metrics is None:
            print(
                "Metrics collection is not enabled. Initialize with "
                "collect_metrics=True."
            )
            return
        self.metrics.print_aggregated_stats()

    def _validate_sequence_lengths(self, sequence_lengths: list[int]) -> None:
        """Validate positive integer costs against the bin capacity."""
        for length in sequence_lengths:
            if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
                raise ValueError("sequence lengths must be positive integers")
            if length > self.bin_capacity:
                raise ValueError(
                    f"Sequence length {length} exceeds bin capacity {self.bin_capacity}"
                )


class FirstFitPacker(SequencePacker):
    """Common placement logic for first-fit algorithm variants."""

    @abstractmethod
    def _prepare_sequences(self, sequence_lengths: list[int]) -> list[tuple[int, int]]:
        """Return ``(cost, source_index)`` items in placement order."""

    def _pack_implementation(self, sequence_lengths: list[int]) -> list[list[int]]:
        bins: list[list[int]] = []
        bin_remaining: list[int] = []
        for length, source_index in self._prepare_sequences(sequence_lengths):
            for bin_index, remaining in enumerate(bin_remaining):
                if remaining >= length:
                    bins[bin_index].append(source_index)
                    bin_remaining[bin_index] -= length
                    break
            else:
                bins.append([source_index])
                bin_remaining.append(self.bin_capacity - length)
        return bins


__all__ = ["FirstFitPacker", "SequencePacker"]
