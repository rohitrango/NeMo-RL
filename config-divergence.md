# Config divergences: NeMo-RL vs the Megatron-LM reference

Recorded while making `tools/compare_task_encoders.py` produce a like-for-like
comparison on `~/data/super-test-blend/cook_subset_v2`.

Three sources of truth are involved, and they do not agree with each other:

  A. reference argparse defaults
       energon-megatron-lm/examples/multimodal/multimodal_args.py
       energon-megatron-lm/megatron/training/arguments.py
  B. the production launch script
       energon-megatron-lm/examples/multimodal/v3p5_super_prod_run/
         sft_from_super35_dualds_4k_1of8_recover_iter366_radio_v4_h_full_generalist.sh
  C. NeMo-RL
       nemo_rl/data/energon/config.py  (library defaults)
       examples/configs/sft_v2_tests/*.yaml  (recipe)

NeMo-RL's stated policy is in nemo_rl/data/energon/config.py:57-63:
"Defaults mirror the Megatron reference argparse values ... except
temporal_patch_size: the Nemotron-Omni production run this encoder targets
passes --video-temporal-patch-size 2, so 2 is the useful default here."

## Result: NeMo-RL library defaults are already correct

Every NeMo-RL task-encoder default matches the reference argparse default.

| option                          | NeMo-RL default | ref argparse | match |
| ------------------------------- | --------------- | ------------ | ----- |
| patch_dim                       | 16              | 16           | yes   |
| video_min_num_frames            | 8               | 8            | yes   |
| video_max_num_frames            | 32              | 32           | yes   |
| video_default_fps               | 2               | 2            | yes   |
| video_frame_temporal_jitter     | False           | False        | yes   |
| video_aug_scale_frames_up       | None            | None         | yes   |
| video_aug_scale_resolution_up   | None            | None         | yes   |
| video_aug_scale_resolution_only | False           | False        | yes   |
| allow_large_videos              | False           | False        | yes   |
| thinking_trace_format           | normalized      | normalized   | yes   |
| relax_thinking_trace_check      | False           | False        | yes   |
| audio_clip_duration_seconds     | 30.0            | 30           | yes   |
| temporal_patch_size             | 2               | 1            | deliberate, documented |

No library default needs to change. Changing them to the launch script's values
would make NeMo-RL diverge from the reference's own defaults and bake one
recipe's choices into the library.

## Divergence 1 (open): recipe does not set three video options

The production launch script sets these; the NeMo-RL recipe leaves them at the
default, so the two stacks sample video differently.

| option                          | NeMo-RL recipe | launch script | script line |
| ------------------------------- | -------------- | ------------- | ----------- |
| video_max_num_frames            | 32 (default)   | 64            | 288, 295    |
| video_aug_scale_frames_up       | None (default) | 4             | 290, 298    |
| video_aug_scale_resolution_only | False (default)| true          | 292, 304    |

Measured effect, 190 rows of cook_subset_v2, before alignment:

    video__output_yt1b_qa_plm_sampled_custom   reference 64 frames, nemo_rl 32
    video__nextqa_subset_nmh5r                 reference 64 frames, nemo_rl 32

14 rows had differing token sequences, 36 had differing total_len. After
setting the three options, tokens / loss_mask / total_len / num_frames match on
all 190 rows.

Fix belongs in the recipe, not the library defaults.

## Divergence 2 (closed): recipe already correct

Set in examples/configs/sft_v2_tests/sft_vlm_nemotron_omni_30B_energon_tp4etp4_v2.yaml:54

    prompt_format: nemotron6-moe        matches launch script line 71/449
    thinking_trace_format: ultra        matches launch script line 450

Both differ from the reference argparse defaults (None and "normalized"), and
the recipe correctly overrides them. No action.

## Divergence 3 (open, unresolved): sequence length

| setting                | NeMo-RL recipe | launch script |
| ---------------------- | -------------- | ------------- |
| max_total_sequence_length / decoder_seq_length | 262144 | 524288 (line 183) |
| packing_seq_length     | 262144         | 524288 (line 184) |
| packing_buffer_size    | 32             | 5000 (line 185)   |
| packing algorithm      | greedy_knapsack| balanced_greedy_knapsack, delta=5 (line 262-265) |

The NeMo-RL recipe comments explain the buffer_size and watchdog choices as
deliberate (worker OOM, cold-cache stalls). The seq-length and knapsack
algorithm differences are not explained and may be intentional for the smaller
test recipe. Not investigated.

## Divergence 4 (not a divergence): settings that come from the HF model config

These are reference CLI flags with no NeMo-RL task-encoder option because
NeMo-RL reads them from the model directory instead.

| reference flag             | value    | NeMo-RL source (config.json) |
| -------------------------- | -------- | ---------------------------- |
| --patch-dim 16             | 16       | patch_size: 16               |
| --img-h / --img-w 512      | 512      | force_image_size: 512        |
| --image-tag-type internvl  | internvl | image_tag_type: "internvl"   |
| --pixel-shuffle            | on       | downsample_ratio: 0.5        |
| --vision-model-type radio  | radio    | vision_config auto_map RADIO |

## Harness-only issues (not product divergences)

Recorded so the same time is not spent again. All were defects in
tools/compare_task_encoders.py or its args file, not in either stack.

1. tokenizer_prompt_format was None in the harvested args, so the reference
   skipped _normalize_thinking_trace entirely (gate at task_encoder.py:889) and
   emitted raw source newlines around </think>. Set to nemotron6-moe.
2. keep_history_thinking was not passed to MegatronMultimodalTokenizer, so the
   chat template's own default (True at chat_template.jinja:61) truncated
   history thinking. The launch script passes --tokenizer-keep-history-thinking
   (line 451). Now passed at compare_task_encoders.py:338.
3. REFERENCE_ARGS_TEMPLATE originally carried invented values
   (video_min_num_frames=1, video_max_num_frames=32, video_temporal_patch_size=1)
   that silently overrode harvested parser defaults. Removed. The constructor
   assertion at task_encoder.py:407 caught one of them; the others were silent.
4. Two tokenizers were in play. Both stacks now use
   nvidia/NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained via
   --shared-tokenizer. Its vocab is byte-identical to the Omni model's (0
   differing ids); only the chat template differed.
5. Normalizer defects, all fixed:
   - reference DEFAULT_IMAGE_TOKEN_INDEX (-200) compared against the real
     <image> id (18); now mapped
   - reference total_len compared against NeMo-RL .length; the comparable field
     is packing_cost (same formula: task_encoder.py:2002 vs
     nemotron_visual.py:1119)
   - image_sizes read params.width/height, which ImageTilingParams does not
     have; now reads num_embeddings

## Remaining known gap in the harness

image_count / image_sizes still differ on 36 video rows: the reference stores
UNGROUPED per-frame params (task_encoder.py:1031) while NeMo-RL stores grouped
tubelets. Embedding widths match exactly (120 on both sides); only the grouping
differs, by exactly temporal_patch_size=2. total_len already agrees because the
reference uses grouped params for sequence length.

## Verification command

    uv run --locked --extra energon tools/compare_task_encoders.py \
      --subset ~/data/super-test-blend/cook_subset_v2/subset.yaml \
      --model-path ~/data/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
      --shared-tokenizer /tmp/shared_tokenizer \
      --nemo-rl-option video_max_num_frames=64 \
      --nemo-rl-option video_aug_scale_frames_up=4 \
      --nemo-rl-option video_aug_scale_resolution_only=true \
      --config examples/configs/sft_v2_tests/vlm_sft-nemotron-omni-30ba3b-4n8g-megatron-tp4etp4-super-test-blend.v1.yaml \
      --reference-root energon-megatron-lm --reference-args /tmp/ref_args.json

Result: 190 rows, 150 identical, 36 differ (image_count/image_sizes grouping
only), 4 reference-only errors on the missing-metadata leaves
reasoning_on__benchfit_qa and reasoning_on__dense_ocr_qa.

================================================================================
FULL SWEEP RESULTS AND TWO FURTHER FINDINGS
================================================================================

Full v2 sweep, 15990 rows, both cookers + both task encoders, packing off.
Run time ~13 min.

    rows                            15990
    identical                       12222
    differ                           3331
    only_reference_raised_encode      369
    only_nemo_rl_raised_cook           31
    only_nemo_rl_raised_encode          6
    both_raised_encode                 26
    both_raised_cook                    5

    final_tokens / final_loss_mask / final_length / total_len   453 rows
    image_sizes                                                3331 rows
    image_count                                                2878 rows

The 190-row sample run reported zero final-sequence differences. The full sweep
found 453. Sampling 2 rows per leaf was not enough.

## Divergence 5 (config): random tiling augmentation

453 rows, concentrated in three leaves:
    reasoning_off__det_2_nlp_coord_clean
    reasoning_off__figureqa_nmh5r
    reasoning_off__TAL_HW_MATH_nmh5r_clean

Symptom: NeMo-RL plans more image embeddings than the reference for the same
image. The height patch dimension always matches; the width is larger by
exactly 32 patches (= 512 px = one tile edge).

    image 599x400   reference patch_size (40, 28)  ->  280 embeddings
                    nemo_rl   patch_size (72, 28)  ->  504 embeddings
    image 567x400   reference (38, 28) -> 266      nemo_rl (70, 28) -> 490
    image 692x400   reference (44, 26) -> 286      nemo_rl (76, 26) -> 494

Cause: tiling augmentation, which is random.

  reference  task_encoder.py:737
      data_augment = sample.__subflavors__.get("data_augment", False) \
                     and not self.is_val
      The harness builds MultiModalTaskEncoder(is_val=True), so augmentation is
      OFF on the reference side.

  nemo_rl    nemotron_visual.py:905
      if data_augment and random.random() < self.tiling_augment_prob:
      There is no is_val gate. tiling_augment_prob defaults to 0.4, so NeMo-RL
      scales tiles up on roughly 40% of rows from augmented leaves.

20 of the 95 leaves carry data_augment: true in subset.yaml, but only image
leaves with a scale-up draw show the difference.

Verification: rerunning the three leaves with --nemo-rl-option
tiling_augment_prob=0.0 gives 12/12 identical on each. That isolates it to
augmentation and nothing else.

This is not a logic divergence. Two independent RNGs cannot agree, so a parity
harness must disable augmentation on both sides. Whether NeMo-RL should grow an
is_val equivalent for validation splits is a separate product question and is
NOT answered here.

## Divergence 6 (product, NeMo-RL only): literal "<image>" in text rejected

6 rows, both in the text branch:
    text__part-00012-002.materialized.jsonl   1 of 200
    text__part-00019-002.materialized.jsonl   5 of 200

NeMo-RL raises at pre-encode; the reference encodes the row without complaint.

    ValueError: Nemotron Omni sample '.../000175' tokenizes to 16 image tokens
    but pre-encoding planned 0 from 0 visual media items. The conversation text
    most likely contains a literal '<image>' substring that is not backed by a
    media entry; the tokenizer maps it to the image token.

The error message is correct. Row 175 of part-00019-002 is a text-only sample
whose content contains 16 literal "<image>" substrings, in prose about a CLI
tool:

    "...when scanning an image, `grype` resolves the image to a specific digest.
     If you use `grype <image>`, it might show something like ..."

The tokenizer maps "<image>" to id 18 because it is a real special token in the
vocabulary. NeMo-RL cross-checks the tokenized image-token count against the
planned visual media count and rejects the mismatch. The reference performs no
such check, so it emits 16 image tokens with no backing media.

Which behaviour is preferable is a judgment call:
  - NeMo-RL fails loudly and drops 6 rows out of 15990 (0.04%).
  - The reference trains on rows whose token stream claims images that do not
    exist.
The check itself looks correct; the open question is whether the fix belongs in
the data (escape the literal) rather than in either loader.

## Remaining known harness gap (unchanged)

image_count / image_sizes differ on video rows only: the reference stores
UNGROUPED per-frame params (task_encoder.py:1031), NeMo-RL stores grouped
tubelets. Embedding widths match exactly. final_tokens and final_length agree on
those rows, so the model sees identical input.

## Rows both stacks reject (agreement, no action)

26 both_raised_encode: malformed thinking traces, same rows, same cause.
    reference  AssertionError: Found sample with 1 <think> tags and 2 </think> tags
    nemo_rl    ValueError: Nemotron assistant turns require exactly one matched
               pair of <think> tags

369 only_reference_raised_encode: the missing-metadata leaves
reasoning_on__benchfit_qa and reasoning_on__dense_ocr_qa, exactly as predicted
by the cooker comparison. NeMo-RL derives the metadata and trains these rows.

31 only_nemo_rl_raised_cook: the empty-conversation rows in
reasoning_on__benchfit_qa. The reference cooks them and dies later at
task_encoder.py:789.
