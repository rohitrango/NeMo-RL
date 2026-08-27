# Cooker parity: NeMo-RL vs the Megatron-LM reference

Comparison of the Energon cookers in
`nemo_rl/data/energon/multimodal/cookers/` against the reference at
`examples/multimodal/data_loading/cookers/` in the Megatron-LM checkout
(`energon-megatron-lm/`).

Reproduce with `tools/compare_cookers.py`. Both cookers receive the same crude
sample, the same stub `FileStore`, and the same stub `CachePool`, so any output
difference is attributable to the cooker. Each side gets its own deep copy of
the sample because the reference mutates `sample['json']` in place.

## Summary

| blend | rows | identical | differ | ref-only error | nrl-only error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cook_subset_v1` | 7807 | 4607 | 3200 | 0 | 0 |
| `cook_subset_v2` | 15990 | 13785 | 2169 | 0 | 31 |

Row counts match `MANIFEST.json` in both blends, so nothing was skipped.

`roles`, `part_structure`, `text`, `media_modalities`, `media_values`, and
`sources` are identical on all 23797 rows across both blends. Every divergence
falls into three buckets.

There is no row in either blend where the reference produces usable output and
NeMo-RL does not.

## Divergence 1 — `train_on_message` (metadata only)

Rows: 3000 in v1, 1600 in v2. Every leaf carrying
`train_only_on_last_assistant_turn: true`.

```
reference   [None, None, None, None, None]
nemo_rl     [False, False, False, False, True]
```

- The reference leaves `Message.loss = None` and applies the mask later in
  `task_encoder.py:739`.
- NeMo-RL's `_apply_last_assistant_mask` (`nemotron.py:449`) writes the flag in
  the cooker.
- The token-level mask still comes from the subflavor in both stacks
  (`nemotron_tokenization.py:112`, `task_encoder.py:739`), so this field is
  carried but not acted on by the NeMo-RL encoder when `loss_mask_mode` is
  unset.

Not verified end to end: the encoders were not run. Treat "inert" as likely,
not proven.

## Divergence 2 — `subflavors["dataset"]` (metadata only)

Rows: 200 in v1, 200 in v2. One leaf per blend, the one whose payload carries a
top-level `dataset` field.

```
key "dataset"   reference: <absent>     nemo_rl: "babyvision"
```

- NeMo-RL's `_sample_keys_with_dataset` lifts the payload's `dataset` field
  into subflavors from every cooker.
- The reference does this only in `cook_conversation`
  (`_basic_sample_keys_with_json_dataset`). The general cookers route through
  `conversation_post_processing`, which ends with plain `basic_sample_keys`.

## Divergence 3 — derived media metadata (behavior)

Rows: 0 in v1, 369 in v2 — `reasoning_on__dense_ocr_qa` (200) and
`reasoning_on__benchfit_qa` (169). Both are `cook: conversation` leaves.

All 369 have the same one-way shape. No conflicting values anywhere.

```
reference_only = []
nemo_rl_only   = [format, height, mode, width]      e.g. JPEG, 562, RGB, 1000
```

### Why

Both ask the store first; `get_media_metadata` fails because these media dirs
have no prepared `.nv-meta`.

- Reference `cook_conversation` (`conversation.py:374-383`) warns once per
  store, leaves `frag.metadata = None`, and never derives.
- NeMo-RL passes `derive_missing_metadata=True`, so `_aux_media`
  (`nemotron.py:301`) does a blocking `cache.get`, opens the image, and reads
  width/height/format/mode.

The reference is inconsistent with itself here: `conversation_base.py:312-318`
derives metadata for the general/jsonl/webdataset cookers. Only the fragment
cooker skips it. NeMo-RL's flag makes the fragment cooker match its siblings.

v1 showed zero of these because all 1600 of its media fragments carry an
explicit `metadata` field; both cookers then skip the lookup and the decode.

### Why NeMo-RL needs the metadata

`nemotron_visual.py:819` treats it as required input, not a hint:

```python
frames.append(_VisualFrameSpec(
    width=_required_metadata_int(ref, "width"),
    height=_required_metadata_int(ref, "height"),
    is_video=False,
))
```

| modality | required keys |
| --- | --- |
| image | `width`, `height` |
| video | `video_duration`, `video_num_frames`, `sampled_num_frames`, `sampled_fps` |
| video_frame | `timestamp` |
| audio | `audio_sample_rate`, `audio_duration` |

The chain is `width/height -> _radio_token_count() -> patch grid ->
visual_embeddings -> packing_cost` (`nemotron_visual.py:1119`). Packing must
know the sequence length before the pixels are decoded, and the media value is
a lazy reference at that point. So either the cooker supplies the dimensions or
the planner has nothing to compute with.

### What the reference does instead

It drops the sample. It does not recover and it spends no extra time.

```
cook_conversation
   metadata = None, value = cache.get_lazy(...)          no file read
        |
preencode_sample, task_encoder.py:951
   self.image_tiling_strategy.compute_params(image_media, ...)
        |
image_processing.py:451 / 1280 / 1541
   img_size = (media.width, media.height)
        |
ImageMedia.width  ->  self.metadata["width"]
        |
   TypeError: 'NoneType' object is not subscriptable
        |
compact_sample_error_handler, dataloader_provider.py:283
   prints "Ignoring error processing sample:" + traceback, returns None
        |
   sample skipped, training continues
```

`compute_params` at line 951 is unconditional for any sample with images.
Grepping `task_encoder.py` and `image_processing.py` for assignments to
`.metadata`, `.width`, or `.height` returns zero hits, so nothing between the
cooker and the tiling strategy fills it.

Verified in isolation:

```
reference frag.metadata : None
frag.width -> TypeError: 'NoneType' object is not subscriptable
```

Not verified end to end: the reference task encoder was not executed. That
needs megatron-core, a tokenizer, and the full args object.

### Cost

| | reference | NeMo-RL |
| --- | --- | --- |
| media read in the cooker | none | 1 blocking read + decode |
| outcome | sample dropped | sample trained |
| signal | one traceback per sample | none |

Measured blocking reads (reference vs NeMo-RL): v1 3744 vs 3744 (no
difference), v2 4510 vs 4884 (374 extra, one per affected media item).

### Impact on v2

Under the reference, these two leaves contribute nothing to training:

| leaf | rows | dropped | cause |
| --- | ---: | ---: | --- |
| `reasoning_on__benchfit_qa` | 200 | 31 | empty `conversation` -> `IndexError` at `task_encoder.py:789` |
| | | 169 | `metadata=None` -> `TypeError` in `compute_params` |
| | | **200/200** | |
| `reasoning_on__dense_ocr_qa` | 200 | **200/200** | `metadata=None` |

Under NeMo-RL, 369 of those 400 rows train.

## Rows both stacks reject

### Empty `conversation` — 31 rows, v2

`reasoning_on__benchfit_qa` holds 31 rows shaped like this:

```json
{"image_path": "...mint_batch_32156.tar/...jpg",
 "image_id": "mint_batch_32156/...",
 "conversation": []}
```

| | reference | NeMo-RL |
| --- | --- | --- |
| where | task encoder, `task_encoder.py:789` | cooker, `nemotron.py:479` |
| error | `IndexError: list index out of range` | `ValueError: Nemotron fragment conversations require a non-empty list.` |

Verified by running the reference cooker: it succeeds with a zero-message
conversation, and `sample.conversation[0]` then raises. There is no length
guard anywhere in `preencode_sample`; the only hits for `conversation[0]` are
lines 789 and 792, both indexing.

Both raise inside the Energon pipeline, so the same error handler decides
skip-or-die for both. NeMo-RL fails earlier with a message that names the
cause.

### Malformed rows — 5 rows, v2

`video__output_short_video_qa_vggsound_custom`. Reference raises `KeyError` x4
and `IndexError` x1; NeMo-RL raises `ValueError` x5. Same verdict, clearer
message.

## Metadata dropped by both

Neither cooker carries these payload keys into the sample.

| key | v1 rows | v2 rows |
| --- | ---: | ---: |
| `id` | 1207 | 2165 |
| `provenance` | 2400 | 800 |
| `image_path` | 0 | 400 |
| `image_id` | 0 | 400 |
| `category` | 7 | 11 |

Per-message keys other than `from`/`value` (or `role`/`content`) are also
dropped silently by both. For the OpenAI schema that includes `tool_calls`,
`name`, `reasoning_content`, and `weight`. An assistant turn carrying only
`tool_calls` with `content: null` becomes a blank `"\n"` turn in both. Neither
blend exercises this — no row carries an extra message key.

## Divergences that exist in the code but neither blend triggers

Read from source, never reached by v1 or v2 data.

| case | reference | NeMo-RL |
| --- | --- | --- |
| aux stores via `aux_data_prefixes` | broken: `conversation.py:492` passes `media_sources=media_sources` instead of `**media_sources`, so `retrieve_media_source` always misses and the sample falls to the local-file check | works |
| media basename with no dot, e.g. `frame001` | `IndexError` on `basename.split('.',1)[1]` | falls back to the whole basename |
| two media tags falling back to the same default extension | reuses the same tar member twice | skips an already-used extension, then falls to the aux/local path |
| media entry is a dict (`{"member": ..., "metadata": ...}`) | `TypeError` in `os.path.basename` | handled by `_descriptor`; carries per-entry metadata the reference cannot parse |
| unknown `from` sender, e.g. `"gpt"` on the fragment schema | passed through, asserts later at `task_encoder.py:838` | mapped or rejected in the cooker |
| user turn with the open-think marker more than once | `content.replace(...)` rewrites every occurrence | `removesuffix(...)` rewrites only the trailing one |
| content part with a falsy-but-present `type` | `part.get("type") or part.get("t")` falls through, part accepted | `part.get("type", part.get("t"))` yields `""`, raises |

Neither blend has any `aux_data_prefixes` entry.

## Two defects in the NeMo-RL cookers

- `nemotron.py:627-631` is unreachable. `_aux_media` either raises or returns a
  non-`None` `SourceInfo` on every path, so the `Media member ... is absent`
  error and the trailing `return source, (), None` can never run.
- The docstring at `nemotron.py:800` says "without opening its media".
  `_primary_media` calls `_open_media` on every member unconditionally.

## Coverage

Verified against real rows:

| cooker | v1 rows | v2 rows |
| --- | ---: | ---: |
| `nemotron_conversation` | 2000 | 9025 |
| `nemotron_general_conversations_jsonl` | 1807 | 2765 |
| `nemotron_general_conversations_webdataset` | 2000 | 400 |
| `nemotron_nano_openai_messages_jsonl` | 2000 | 3800 |

Not exercised — no rows in either blend:

- `nemotron_general_conversations_jsonl_explicit_loss_v1`
- `nemotron_granary_english_webdataset`
- `nemotron_granary_english_jsonl`
- `nemotron_nano_openai_messages_offline_packed_jsonl`
- `nemotron_audio_conversation_jsonl`
- `nemotron_omcat_legacy_conversation_monolithic`

All six are already wired into `tools/compare_cookers.py`; point it at a subset
containing them and they run with no code change.

`generic_conversation` (`cookers/generic.py`) has no reference counterpart and
is outside this comparison.

No audio path was exercised. Neither blend contains audio: `cook_subset_v1`
holds 3281 png, 639 jpg, 254 mp4, 3 jpeg, 1 webp and zero audio containers, and
no row references `<audio>`, `<sound>`, `<speech>`, or `<video-sound>`. The mp4
files do carry aac/opus tracks, but nothing routes them as audio.

## Method and its limits

- Stub `FileStore` whose `get_media_metadata` always fails, matching an exported
  subset with no prepared `.nv-meta`. Both cookers then take their
  derive-from-media path.
- Stub `CachePool` that decodes the real file on `get`, so derived width/height
  and AV metadata are real. `get_lazy` and `to_cache` return comparison markers.
- Reference media timing fields (`start_time`, `end_time`, `timestamp`,
  `frame_index`, `sample_index`) are folded into the metadata tuple, and
  null-valued metadata entries are stripped on both sides. NeMo-RL copies a
  timing key whenever it is present, so `start_time: null` in the source would
  otherwise read as a difference. `reasoning_off__robovqa` has exactly this
  shape.

What this does not cover:

- Real Energon `FileStore` / `CachePool`. Lazy values are compared as markers,
  not resolved.
- The prepared-metadata branch, since the stub always fails.
- The task encoders. Use `tools/compare_task_encoders.py` for that.
- Any dataset carrying audio, `aux_data_prefixes`, or the six untested cookers.
