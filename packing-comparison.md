# Packing parity: NeMo-RL vs the Megatron-LM reference

Third stage of the Energon parity work. The first two stages compared cookers
(`cooker-comparison.md`) and task encoders through pre- and post-encoding. Both
found the token ids and loss masks identical, with packing disabled. This stage
turns packing on.

Reproduce with `tools/compare_packing.py`.

## Summary

Everything matches once both stacks are configured the same way. One real
defect was found on the way: the recipe's `sequence_length_pad_multiple` was 8
where the reference computes 16.

| check | result |
| --- | --- |
| cost per sample | identical, 112/112 |
| `greedy_knapsack` bins | identical membership and fills |
| `balanced_greedy_knapsack` (delta=5) | identical membership and fills |
| oversize criteria | same predicate on both sides |
| tie-breaking | differs internally, no effect on output |
| `get_padding` vs `sequence_length_pad_multiple` | **8 was wrong; 16 is correct** |
| `pack_selected_samples` / `cu_seqlens` | identical |

## What the script compares, and why it is split

Two questions that a single "do the packs match" check would conflate:

1. **Do the two stacks assign the same COST to each sample?**
2. **Given identical costs and capacity, do the two implementations produce
   the same BINS?**

For (2) the script feeds *both* algorithms the reference's cost list, so a
membership difference is provably the algorithm and not the input.

Cost inputs differ in where the padding comes from:

    reference  select_samples_to_pack uses sample.total_len_padded
               = total_len + get_padding(...), applied only when
               cp_size > 1 or sequence_parallel
    nemo_rl    packing/sft.py _aligned_packing_cost
               = round_up(packing_cost, sequence_length_pad_multiple)

## Finding 1: balanced_greedy_knapsack was missing from NeMo-RL

The production launch script selects it:

    USE_BUCKETING=${USE_BUCKETING:-0}          # line 164, so the else branch
    --packing-knapsack-algorithm balanced_greedy_knapsack       # line 264
    --packing-algorithm-parameters balanced_knapsack_delta=5    # line 265

NeMo-RL offered only `concatenative`, `first_fit_decreasing`,
`first_fit_shuffle`, `modified_first_fit_decreasing`, `greedy_knapsack`. The
recipe therefore used `greedy_knapsack` -- a different algorithm from the one
the reference run uses.

The two are not interchangeable. On the same 112 samples:

    greedy_knapsack            7 bins   fills one bin to the brim, then opens the next
    balanced_greedy_knapsack  12 bins   sorts descending, always fills the least-full bin

Ported as `nemo_rl/data/packing/balanced_greedy_knapsack.py`, following
`knapsacks.py:66` at Megatron-LM commit
6822175d92a40e0528be905aee50f5930cfa0c98. Registered in
`nemo_rl/data/packing/factory.py` and in the Energon `PACKING_REGISTRY`.

Two deliberate deviations, documented in the module docstring:

* **Oversized items.** The reference prints a warning and silently drops them.
  `SequencePacker.pack` validates first and raises, so an oversized item never
  reaches the implementation. The skip branch is kept for parity when
  `_pack_implementation` is called directly.
* **Empty bins.** The reference returns them -- the `+delta` pre-allocation
  guarantees some. Preserved so bin counts match.

Verified against the reference on 11 synthetic cases (ties, 200 random costs,
all-identical costs, one dominant item) at delta 5 and 20: identical bin counts
and membership in every case.

Plumbing needed to reach it from a recipe:

* `EnergonPackingOptions` is `extra="forbid"`, so `balanced_knapsack_delta`
  needed a real field.
* `build_packing_hooks` called `packer_type(options.max_sequence_length)`
  positionally; it now forwards algorithm-specific options when set.
* The Energon `PACKING_REGISTRY` is separate from `factory.py` and needed its
  own entry.

## Finding 2: the pad multiple was 8, should be 16

The recipe carried:

    sequence_length_pad_multiple: 8  # 2 * context_parallel_size(4)

The arithmetic is right; the branch is wrong. `get_padding`
(`megatron/core/models/multimodal/context_parallel.py:46`):

```python
padding_factor = 1
if has_sp and cp_size > 1:
    padding_factor = max(tp_size * cp_size, cp_size * 2)
elif cp_size > 1:
    padding_factor = cp_size * 2
elif has_sp:
    padding_factor = tp_size
```

| condition | factor | this recipe |
| --- | --- | --- |
| `has_sp and cp_size > 1` | `max(tp*cp, cp*2)` = **16** | applies |
| `cp_size > 1` only | `cp*2` = 8 | the comment's formula |
| `has_sp` only | `tp` = 4 | |

The recipe sets `tensor_model_parallel_size: 4`, `context_parallel_size: 4`,
`sequence_parallel: true`, so `max(4*4, 4*2) = 16`. The `2 * cp_size` value is
correct only when sequence parallel is off.

Measured over 1..4999:

    get_padding(n) == round_up(n, 16)   4999/4999
    get_padding(n) == round_up(n,  8)   2496/4999

Divisibility of the padded totals: always divisible by 2, 4, 8 and 16; not by
32 or 64.

Note the reference's own comment says "multiple of `tp_size * cp_size * 2`"
while the code computes `max(tp_size * cp_size, cp_size * 2)`. Those differ
(32 vs 16 here); the code is what runs.

With `fp8_enabled` the factor grows again (`fp8_factor = 16 * padding_factor`),
which is a further argument against a hardcoded constant.

### Why it matters

Not cosmetic. Commit b2b14e13 records the failure mode when this value is
wrong:

    ValueError: Every prepacked padded source length must be divisible by
                2 * context_parallel_size (8).

raised inside the first forward pass, when Megatron slices each padded
sub-sequence across CP ranks. The existing guard checks `2 * cp_size` = 8; 16
satisfies it, but that guard would not have caught 8 being too small for a
recipe with sequence parallel and TP > 1.

Recipe now sets 16, with the derivation in a comment. `262144 % 16 == 0`, so
the `EnergonPackingOptions` alignment validator still passes.

A constant is still the wrong shape for this: the correct multiple is a
function of CP, TP, SP and fp8. 16 is right for this recipe and wrong for
another.

## Verification, both stacks at the recipe's parallelism

Reference run with `context_parallel_size: 4`, `tensor_model_parallel_size: 4`,
`sequence_parallel: true`; NeMo-RL with `sequence_length_pad_multiple: 16`.
112 text samples.

    cost agreement       112/112 matching

    packing
      algorithm          balanced_greedy_knapsack  delta=5
      tied costs         27/112 (98 distinct)
      bins               reference 12   nemo_rl 12
      membership         IDENTICAL
      bin fills          IDENTICAL

    pack construction
      per-sample padded  IDENTICAL
      cu_seqlens         IDENTICAL
      pack total         reference 190432   nemo_rl 190432
      all lengths % 8    reference True   nemo_rl True
      all lengths % 16   reference True   nemo_rl True

Pack totals agree to the token. At pad multiple 8 the NeMo-RL total was 190384
against the reference's 190432 -- a 48-token shortfall across 12 samples.

Ties only appeared once padding was active: 0 tied costs unpadded, 27 at
multiple 16, because padding collapses nearby lengths onto the same value.
Membership stayed identical, which answers the tie-break question on real data.

### cu_seqlens shapes

The two carry the same information differently:

    reference  cu_lengths / cu_lengths_padded -- running offsets, leading 0
               (task_encoder.py:1418)
    nemo_rl    source_lengths / source_padded_lengths -- per-sample values
               (packing/sft.py:88)

`PackedSFTSample` has no `cu_seqlens` field; the consumer takes the cumulative
sum. The comparison therefore compares `accumulate(padded_lengths)` against the
reference's stored offsets.

## Tie-breaking: a difference that does not propagate

The sort keys differ:

    reference   sorted(zip(item_sizes, samples), key=lambda x: x[0])
                stable sort + rightmost fit -> LAST of a tied group
    nemo_rl     sorted((length, -source_index, source_index) ...)
                -> SMALLEST source index of that group

The outcome is the same. When costs are equal it does not matter which of the
equal items is taken: any choice removes the same capacity and leaves an
equivalent remaining multiset. Confirmed on a synthetic case with 16 of 18
samples tied, and on real data with 27/112 tied.

NeMo-RL's `-source_index` is about determinism, not correctness.

## Oversize handling: same predicate

    reference  knapsacks.py:41   max(costs) > capacity -> raise
    nemo_rl    packing/base.py:141  any(cost > capacity) -> raise
                                    also rejects non-int, bool, <= 0

Both unreachable at the recipe's capacity, because each stack truncates before
packing. One asymmetry, untested: the reference truncates against
`decoder_seq_length` but packs against `packing_seq_length`, so a recipe with
`packing_seq_length < decoder_seq_length` could produce an oversized sample.
NeMo-RL truncates against the value it packs against. Both are 262144 here.

## What is NOT covered

* **Text leaves only.** 112 samples. Image and video leaves have not been
  through the CP=4 packing comparison.
* **`buffer_size` still differs.** 32 in the recipe against the reference's
  5000. Deliberate -- b2b14e13 documents the worker OOM at larger buffers with
  162MB clips -- but it means the two pack over different candidate windows, so
  which samples are even eligible for the same pack differs.
* **`select_samples_to_pack` wrappers.** The script calls the algorithms
  directly, so the reference's `shuffle_packed_samples` and NeMo-RL's
  `_adjust_bin_count` / `min_bin_count` / `bin_count_multiple` are unexercised.
* **`batch()`** on either side.
* **Streaming behaviour.** Packing consumes a buffer; the script feeds one flat
  ordered list.
* **Audio.** Neither blend contains any.

## Harness pitfalls, recorded

Two false findings came from harness configuration, not from either stack:

1. The first `cu_seqlens` run reported `per-sample padded: DIFFERS` and
   `all lengths % 8 == 0: reference False`. The reference was running at CP=1
   with sequence parallel off, where `_pad_for_context_parallel_and_fp8` is a
   no-op, so it emitted unpadded lengths. A reference that never pads cannot
   fail an alignment check. Fixed by running it at the recipe's parallelism.
2. Comparing at capacity 262144 put all 152 image samples in a single bin,
   which exercises no packing decision at all. "Membership identical" there is
   arithmetic, not evidence. Capacity has to be low enough to force contention.

## Commands

    # algorithm isolated (no padding on either side)
    python tools/compare_packing.py --limit 6 --leaf text__ \
      --algorithm balanced_greedy_knapsack --delta 5 --pad-multiple 1

    # both stacks at the recipe's parallelism
    python tools/compare_packing.py --limit 6 --leaf text__ \
      --algorithm balanced_greedy_knapsack --delta 5 --pad-multiple 16 \
      --reference-args /tmp/ref_args_cp4.json

`/tmp/ref_args_cp4.json` is `tools/gen_reference_args.py` output with
`context_parallel_size: 4`, `tensor_model_parallel_size: 4`,
`sequence_parallel: true`.
