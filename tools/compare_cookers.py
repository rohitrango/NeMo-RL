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

"""Run the NeMo-RL cookers and the Megatron-LM reference cookers over one
Energon subset and report where they disagree.

Both cookers receive the same crude sample, the same stub ``FileStore``, and the
same stub ``CachePool``, so any difference in the output is attributable to the
cooker. Each cooker gets its own deep copy of the sample because the reference
implementation mutates ``sample['json']`` in place.

The two output types are normalized to a common form before the diff:

    {roles, parts (ordered text/media), media[modality, metadata, value],
     loss_flags, subflavors, sources}

Reference media timing fields (``start_time``, ``timestamp``, ``frame_index``,
``sample_index``) are folded into the metadata tuple when they are not ``None``,
because NeMo-RL stores them there. A source fragment carrying an explicit null
timing value is therefore reported as equal.

Usage (inside the container):

    uv run --locked --extra energon tools/compare_cookers.py \
        --subset ~/data/super-test-blend/cook_subset_v1/subset.yaml \
        --reference-root energon-megatron-lm \
        --limit 25
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

# Reference modules searched for a cooker by name, in order.
REFERENCE_MODULES = (
    "examples.multimodal.data_loading.cookers.conversation",
    "examples.multimodal.data_loading.cookers.granary",
    "examples.multimodal.data_loading.cookers.audio_conversation",
    "examples.multimodal.data_loading.cookers.omcat_legacy_audio_conversation",
)
# NeMo-RL modules searched for a cooker by name, in order.
NEMO_RL_MODULES = (
    "nemo_rl.data.energon.multimodal.cookers.nemotron",
    "nemo_rl.data.energon.multimodal.cookers.nemotron_legacy",
)

_MEDIA = frozenset({"media_source", "aux"})
_FULL = frozenset({"primary", "media_source", "aux"})

# Cook subflavor -> (reference name, NeMo-RL name, store kwargs both accept).
COOK_PAIRS = {
    "conversation": ("cook_conversation", "cook_nemotron_conversation", _MEDIA),
    "general_conversations_jsonl": (
        "cook_general_conversations_jsonl",
        "cook_general_conversations_jsonl",
        _FULL,
    ),
    "general_conversations_webdataset": (
        "cook_general_conversations_webdataset",
        "cook_general_conversations_webdataset",
        _FULL,
    ),
    "general_conversations_jsonl_explicit_loss_v1": (
        "cook_general_conversations_jsonl_explicit_loss_v1",
        "cook_general_conversations_jsonl_explicit_loss_v1",
        _FULL,
    ),
    "openai_messages_jsonl": (
        "cook_openai_messages_jsonl",
        "cook_nano_openai_messages_jsonl",
        frozenset({"media_source"}),
    ),
    "openai_messages_offline_packed_jsonl": (
        "cook_openai_messages_offline_packed_jsonl",
        "cook_nano_openai_messages_offline_packed_jsonl",
        frozenset({"media_source"}),
    ),
    "granary_english_webdataset": (
        "cook_granary_english_webdataset",
        "cook_granary_english_webdataset",
        _FULL,
    ),
    "granary_english_jsonl": (
        "cook_granary_english_jsonl",
        "cook_granary_english_jsonl",
        _FULL,
    ),
    "audio_conversation": (
        "cook_audio_conversation",
        "cook_audio_conversation_jsonl",
        _FULL,
    ),
    "omcat_legacy_conversation_monolithic": (
        "cook_omcat_legacy_conversation_monolithic",
        "cook_omcat_legacy_conversation_monolithic",
        frozenset({"primary"}),
    ),
}

# Payload keys each cook family reads. Anything else is dropped by both.
CONSUMED_KEYS = {
    "conversation": {"conversation", "dataset"},
    "openai": {"messages", "dataset"},
    "general": {
        "conversations",
        "dataset",
        "image",
        "images",
        "video",
        "videos",
        "audio",
        "audios",
        "sound",
        "speech",
        "speeches",
        "video-sound",
    },
    "granary": {"audio_filepath", "text", "dataset"},
}

DIFF_CATEGORIES = (
    "roles",
    "part_structure",
    "text",
    "media_modalities",
    "media_metadata",
    "media_values",
    "loss_flags",
    "subflavors",
    "sources",
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _consumed_family(cook: str) -> str:
    if cook == "conversation":
        return "conversation"
    if cook.startswith("openai_messages"):
        return "openai"
    if cook.startswith("granary"):
        return "granary"
    return "general"


class StubStore:
    """Filesystem-backed stand-in for an Energon ``FileStore``.

    ``get_media_metadata`` always fails, which matches an exported subset with no
    prepared ``.nv-meta``. Both cookers then take their derive-from-media path.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def get_path(self) -> str:
        return str(self.root)

    def get_media_metadata(self, path: str) -> Any:
        raise FileNotFoundError(f"no prepared media metadata for {path!r}")

    def resolve(self, path: str) -> Path:
        return self.root / path


class StubLazy:
    """Stand-in for an Energon ``Lazy`` value.

    Both task encoders call ``.get(sample)`` during post-encoding, and the
    reference additionally uses the value as a dict key to group frames of the
    same video (task_encoder.py:1732), so it must hash by identity of the
    underlying file.
    """

    __slots__ = ("store", "path")

    def __init__(self, store: "StubStore", path: str) -> None:
        self.store = store
        self.path = path

    def get(self, sample: Any = None) -> Any:
        return _open_file(self.store.resolve(self.path))

    def descriptor(self) -> tuple[str, str, str]:
        return ("lazy", self.store.get_path(), self.path)

    def __hash__(self) -> int:
        return hash(self.descriptor())

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, StubLazy) and self.descriptor() == other.descriptor()

    def __repr__(self) -> str:
        return f"StubLazy{self.descriptor()}"


class StubCache:
    """Stand-in for an Energon ``CachePool`` that records what was requested."""

    def __init__(self) -> None:
        self.get_calls = 0
        self.lazy_calls = 0
        self.to_cache_calls = 0

    def get(self, store: StubStore, path: str) -> Any:
        self.get_calls += 1
        return _open_file(store.resolve(path))

    def get_lazy(self, store: StubStore, path: str) -> "StubLazy":
        self.lazy_calls += 1
        return StubLazy(store, path)

    def to_cache(self, value: Any, key: str) -> tuple[str, str]:
        self.to_cache_calls += 1
        return ("cached", key)


def _open_file(path: Path) -> Any:
    """Decode a media file the way an Energon cache pool would."""
    from megatron.energon.av import AVDecoder
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(f"missing media file {str(path)!r}")
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        image = Image.open(path)
        image.load()
        return image
    return AVDecoder(io.BytesIO(path.read_bytes()))


def _metadata_values(metadata: object) -> Mapping[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return metadata
    if is_dataclass(metadata) and not isinstance(metadata, type):
        return asdict(metadata)
    return vars(metadata)


def _drop_nulls(items) -> tuple[tuple[str, Any], ...]:
    """Explicit nulls carry no information in either representation."""
    return tuple(sorted((k, v) for k, v in items if v is not None))


def _norm_metadata(metadata: object) -> tuple[tuple[str, Any], ...] | None:
    if metadata is None:
        return None
    if isinstance(metadata, tuple):
        return _drop_nulls(metadata)
    return _drop_nulls(_metadata_values(metadata).items())


def _value_repr(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple) and value and value[0] in ("lazy", "cached"):
        return value
    if isinstance(value, str):
        return ("str", value)
    return ("obj", type(value).__name__)


def _reference_timing(fragment: Any) -> dict[str, Any]:
    keys = ("start_time", "end_time", "timestamp", "frame_index", "sample_index")
    return {
        key: getattr(fragment, key)
        for key in keys
        if getattr(fragment, key, None) is not None
    }


def _norm_sources(cooked: Any) -> list[tuple[str, str]]:
    return [
        (str(source.dataset_path), str(source.index))
        for source in (cooked.__sources__ or ())
    ]


def normalize_reference(cooked: Any) -> dict[str, Any]:
    from examples.multimodal.data_loading.conversation_sample import ConversationSample

    reverse = ConversationSample.__MEDIA_TYPES_REVERSE__
    messages: list[dict[str, Any]] = []
    media: list[dict[str, Any]] = []
    for message in cooked.conversation:
        parts: list[tuple[Any, ...]] = []
        for fragment in message.fragments:
            if isinstance(fragment, str):
                parts.append(("text", fragment))
                continue
            modality = reverse[type(fragment)]
            values = dict(_metadata_values(fragment.metadata))
            values.update(_reference_timing(fragment))
            media.append(
                {
                    "modality": modality,
                    "metadata": (
                        None if fragment.metadata is None and not values
                        else _drop_nulls(values.items())
                    ),
                    "value": _value_repr(fragment.value),
                }
            )
            parts.append(("media", modality, len(media) - 1))
        messages.append({"role": message.sender, "parts": parts})
    return {
        "roles": [message["role"] for message in messages],
        "parts": [message["parts"] for message in messages],
        "media": media,
        "loss_flags": [message.loss for message in cooked.conversation],
        "subflavors": dict(cooked.__subflavors__ or {}),
        "sources": _norm_sources(cooked),
    }


def normalize_nemo_rl(cooked: Any) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in cooked.messages:
        parts: list[tuple[Any, ...]] = []
        for part in message["content"]:
            if part["type"] == "text":
                parts.append(("text", part["text"]))
            else:
                parts.append(("media", part["type"], part["media_index"]))
        messages.append({"role": message["role"], "parts": parts})
    return {
        "roles": [message["role"] for message in messages],
        "parts": [message["parts"] for message in messages],
        "media": [
            {
                "modality": ref.modality,
                "metadata": _norm_metadata(ref.metadata),
                "value": _value_repr(ref.value),
            }
            for ref in cooked.media
        ],
        "loss_flags": [
            message.get("train_on_message") for message in cooked.messages
        ],
        "subflavors": dict(cooked.__subflavors__ or {}),
        "sources": _norm_sources(cooked),
    }


def _text_only(parts: list[list[tuple[Any, ...]]]) -> list[str]:
    return [
        "".join(part[1] for part in message if part[0] == "text")
        for message in parts
    ]


def _structure_only(parts: list[list[tuple[Any, ...]]]) -> list[list[Any]]:
    return [
        [part[0] if part[0] == "text" else (part[1], part[2]) for part in message]
        for message in parts
    ]


def classify_diff(ref: dict[str, Any], nrl: dict[str, Any]) -> list[str]:
    """Return the sorted diff categories that differ between the two outputs."""
    found: list[str] = []
    if ref["roles"] != nrl["roles"]:
        found.append("roles")
    if _structure_only(ref["parts"]) != _structure_only(nrl["parts"]):
        found.append("part_structure")
    if _text_only(ref["parts"]) != _text_only(nrl["parts"]):
        found.append("text")
    if [m["modality"] for m in ref["media"]] != [
        m["modality"] for m in nrl["media"]
    ]:
        found.append("media_modalities")
    if [m["metadata"] for m in ref["media"]] != [m["metadata"] for m in nrl["media"]]:
        found.append("media_metadata")
    if [m["value"] for m in ref["media"]] != [m["value"] for m in nrl["media"]]:
        found.append("media_values")
    if ref["loss_flags"] != nrl["loss_flags"]:
        found.append("loss_flags")
    if ref["subflavors"] != nrl["subflavors"]:
        found.append("subflavors")
    if ref["sources"] != nrl["sources"]:
        found.append("sources")
    return found


def _first_field_example(
    ref: dict[str, Any], nrl: dict[str, Any], category: str
) -> dict[str, Any]:
    if category == "text":
        for index, (a, b) in enumerate(
            zip(_text_only(ref["parts"]), _text_only(nrl["parts"]))
        ):
            if a != b:
                return {"message_index": index, "reference": a[:400], "nemo_rl": b[:400]}
    if category == "subflavors":
        keys = set(ref["subflavors"]) | set(nrl["subflavors"])
        return {
            key: {
                "reference": ref["subflavors"].get(key, "<absent>"),
                "nemo_rl": nrl["subflavors"].get(key, "<absent>"),
            }
            for key in sorted(keys)
            if ref["subflavors"].get(key) != nrl["subflavors"].get(key)
        }
    if category in ("media_metadata", "media_values", "media_modalities"):
        field = category.replace("media_", "")
        field = "modality" if field == "modalities" else field.rstrip("s")
        field = {"metadata": "metadata", "value": "value", "modality": "modality"}[field]
        for index in range(max(len(ref["media"]), len(nrl["media"]))):
            left = ref["media"][index].get(field) if index < len(ref["media"]) else "<absent>"
            right = nrl["media"][index].get(field) if index < len(nrl["media"]) else "<absent>"
            if left == right:
                continue
            detail: dict[str, Any] = {"media_index": index}
            if field == "metadata":
                left_map = dict(left or ())
                right_map = dict(right or ())
                detail["only_reference"] = {
                    key: value for key, value in left_map.items()
                    if right_map.get(key, "<absent>") != value
                }
                detail["only_nemo_rl"] = {
                    key: value for key, value in right_map.items()
                    if left_map.get(key, "<absent>") != value
                }
            else:
                detail["reference"] = left
                detail["nemo_rl"] = right
            return detail
    return {
        "reference": str(ref.get(category, ref))[:400],
        "nemo_rl": str(nrl.get(category, nrl))[:400],
    }


def load_leaves(subset_path: Path, split: str) -> list[dict[str, Any]]:
    spec = yaml.safe_load(subset_path.read_text())
    root = subset_path.parent
    entries = spec["splits"][split]
    blend = entries.get("blend_epochized") or entries.get("blend")
    leaves = []
    for entry in blend:
        subflavors = dict(entry.get("subflavors") or {})
        aux = entry.get("aux") or {}
        media_source = aux.get("media_source")
        if isinstance(media_source, str):
            media_source = media_source.split("://", 1)[-1]
        leaves.append(
            {
                "name": PurePosixPath(entry["path"]).stem,
                "cook": subflavors.get("cook"),
                "jsonl": root / entry["path"],
                "media_root": None if media_source is None else root / media_source,
                "subflavors": subflavors,
                "aux": {
                    key: root / str(value).split("://", 1)[-1]
                    for key, value in aux.items()
                    if key != "media_source"
                },
            }
        )
    return leaves


def iter_rows(path: Path, limit: int):
    """Yield (index, payload) for each readable JSONL row."""
    unreadable = 0
    index = 0
    with path.open("rb") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            if limit and index >= limit:
                break
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                unreadable += 1
                continue
            if not isinstance(payload, dict):
                unreadable += 1
                continue
            yield index, payload
            index += 1
    yield None, unreadable


def build_sample(leaf: dict[str, Any], index: int, payload: dict[str, Any]) -> dict:
    key = f"{leaf['name']}/{index:06d}"
    return {
        "__key__": key,
        "__restore_key__": (key,),
        "__subflavors__": dict(leaf["subflavors"]),
        "__sources__": (),
        "json": payload,
    }


def run(args: argparse.Namespace) -> int:
    # tools/ is sys.path[0] when run as a script, so add the repo root for nemo_rl.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(args.reference_root).expanduser().resolve()))
    try:
        ref_modules = [import_module(name) for name in REFERENCE_MODULES]
    except ImportError as error:
        print(f"cannot import the reference cookers from {args.reference_root}: {error}")
        return 2
    try:
        nrl_modules = [import_module(name) for name in NEMO_RL_MODULES]
    except ImportError as error:
        print(
            f"cannot import the NeMo-RL cookers: {error}\n"
            "Run from the repository root, or install the repo, in an environment "
            "with megatron-energon[av-decode], torch, pillow, pydantic, and pyyaml."
        )
        return 2

    def lookup(modules, name: str):
        for module in modules:
            found = getattr(module, name, None)
            if found is not None:
                return found
        return None

    logging.disable(logging.WARNING)

    leaves = load_leaves(Path(args.subset).expanduser(), args.split)
    if args.leaf:
        leaves = [leaf for leaf in leaves if args.leaf in leaf["name"]]

    per_leaf: dict[str, Counter] = defaultdict(Counter)
    diff_categories: Counter = Counter()
    error_pairs: Counter = Counter()
    dropped_keys: Counter = Counter()
    skipped_cooks: Counter = Counter()
    examples: list[dict[str, Any]] = []
    cache_stats = Counter()

    for leaf in leaves:
        cook = leaf["cook"]
        pair = COOK_PAIRS.get(cook)
        if pair is None:
            skipped_cooks[str(cook)] += 1
            continue
        ref_name, nrl_name, accepts = pair
        ref_fn = lookup(ref_modules, ref_name)
        nrl_fn = lookup(nrl_modules, nrl_name)
        if ref_fn is None or nrl_fn is None:
            skipped_cooks[f"{cook} (missing {ref_name}/{nrl_name})"] += 1
            continue

        stats = per_leaf[leaf["name"]]
        stats["cook_" + cook] = 0
        media_store = None if leaf["media_root"] is None else StubStore(leaf["media_root"])
        primary_store = StubStore(leaf["jsonl"].parent)
        aux_stores = {key: StubStore(path) for key, path in leaf["aux"].items()}
        consumed = CONSUMED_KEYS[_consumed_family(cook)]

        for index, payload in iter_rows(leaf["jsonl"], args.limit):
            if index is None:
                stats["source_unreadable"] += payload
                break
            stats["rows"] += 1
            for key in set(payload) - consumed:
                dropped_keys[key] += 1

            ref_cache = StubCache()
            nrl_cache = StubCache()
            kwargs: dict[str, Any] = {}
            if "media_source" in accepts:
                kwargs["media_source"] = media_store
            if "aux" in accepts:
                kwargs.update(aux_stores)
            if "primary" in accepts:
                kwargs["primary"] = primary_store

            sink = io.StringIO()
            ref_out = ref_err = nrl_out = nrl_err = None
            with contextlib.redirect_stdout(sink):
                try:
                    ref_out = ref_fn(
                        build_sample(leaf, index, copy.deepcopy(payload)),
                        cache=ref_cache,
                        **kwargs,
                    )
                except Exception as error:  # noqa: BLE001 - parity comparison
                    ref_err = f"{type(error).__name__}: {error}"
                try:
                    nrl_out = nrl_fn(
                        build_sample(leaf, index, copy.deepcopy(payload)),
                        cache=nrl_cache,
                        **kwargs,
                    )
                except Exception as error:  # noqa: BLE001 - parity comparison
                    nrl_err = f"{type(error).__name__}: {error}"

            cache_stats["ref_blocking_get"] += ref_cache.get_calls
            cache_stats["nrl_blocking_get"] += nrl_cache.get_calls
            cache_stats["ref_lazy"] += ref_cache.lazy_calls
            cache_stats["nrl_lazy"] += nrl_cache.lazy_calls

            if ref_err and nrl_err:
                stats["both_raised"] += 1
                error_pairs[(ref_err.split(":")[0], nrl_err.split(":")[0])] += 1
                continue
            if ref_err:
                stats["only_reference_raised"] += 1
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "leaf": leaf["name"],
                            "row": index,
                            "kind": "only_reference_raised",
                            "reference_error": ref_err[:300],
                        }
                    )
                continue
            if nrl_err:
                stats["only_nemo_rl_raised"] += 1
                if len(examples) < args.max_examples:
                    examples.append(
                        {
                            "leaf": leaf["name"],
                            "row": index,
                            "kind": "only_nemo_rl_raised",
                            "nemo_rl_error": nrl_err[:300],
                        }
                    )
                continue

            ref_norm = normalize_reference(ref_out)
            nrl_norm = normalize_nemo_rl(nrl_out)
            found = classify_diff(ref_norm, nrl_norm)
            if not found:
                stats["identical"] += 1
                continue
            stats["differ"] += 1
            for category in found:
                diff_categories[category] += 1
            if len(examples) < args.max_examples:
                examples.append(
                    {
                        "leaf": leaf["name"],
                        "row": index,
                        "kind": "differ",
                        "categories": found,
                        "detail": {
                            category: _first_field_example(ref_norm, nrl_norm, category)
                            for category in found
                        },
                    }
                )

    report(per_leaf, diff_categories, error_pairs, dropped_keys, skipped_cooks,
           cache_stats, examples)
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "per_leaf": {name: dict(counter) for name, counter in per_leaf.items()},
                    "diff_categories": dict(diff_categories),
                    "error_pairs": {f"{a} / {b}": n for (a, b), n in error_pairs.items()},
                    "dropped_payload_keys": dict(dropped_keys),
                    "skipped_cooks": dict(skipped_cooks),
                    "cache_stats": dict(cache_stats),
                    "examples": examples,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote {args.json}")

    totals = Counter()
    for counter in per_leaf.values():
        totals.update(counter)
    return 1 if (totals["differ"] or totals["only_reference_raised"]
                 or totals["only_nemo_rl_raised"]) else 0


def report(per_leaf, diff_categories, error_pairs, dropped_keys, skipped_cooks,
           cache_stats, examples) -> None:
    columns = ("rows", "identical", "differ", "both_raised",
               "only_reference_raised", "only_nemo_rl_raised", "source_unreadable")
    width = max((len(name) for name in per_leaf), default=10)
    print("=" * (width + 76))
    print(f"{'leaf':<{width}}  " + "  ".join(f"{c[:9]:>9}" for c in columns))
    print("=" * (width + 76))
    totals = Counter()
    for name in sorted(per_leaf):
        counter = per_leaf[name]
        totals.update(counter)
        print(f"{name:<{width}}  " + "  ".join(f"{counter[c]:>9}" for c in columns))
    print("-" * (width + 76))
    print(f"{'TOTAL':<{width}}  " + "  ".join(f"{totals[c]:>9}" for c in columns))

    print("\n--- diff categories (rows affected; a row can hit several) ---")
    if diff_categories:
        for category in DIFF_CATEGORIES:
            if diff_categories[category]:
                print(f"  {category:<18} {diff_categories[category]}")
    else:
        print("  none")

    print("\n--- rows where both raised (exception type pairs) ---")
    if error_pairs:
        for (ref_type, nrl_type), count in error_pairs.most_common(10):
            print(f"  {count:>6}  reference={ref_type:<24} nemo_rl={nrl_type}")
    else:
        print("  none")

    print("\n--- payload keys dropped by BOTH cookers ---")
    if dropped_keys:
        for key, count in dropped_keys.most_common(20):
            print(f"  {count:>6}  {key}")
    else:
        print("  none")

    print("\n--- media access (stub cache call counts) ---")
    for key in ("ref_blocking_get", "nrl_blocking_get", "ref_lazy", "nrl_lazy"):
        print(f"  {key:<18} {cache_stats[key]}")

    if skipped_cooks:
        print("\n--- leaves skipped (no cooker pair) ---")
        for cook, count in skipped_cooks.most_common():
            print(f"  {count:>6}  {cook}")

    if examples:
        print("\n--- examples ---")
        for example in examples:
            print(json.dumps(example, indent=2, default=str)[:1800])
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default="~/data/super-test-blend/cook_subset_v1/subset.yaml",
        help="Path to the exported subset.yaml.",
    )
    parser.add_argument(
        "--reference-root",
        default="energon-megatron-lm",
        help="Megatron-LM checkout holding examples/multimodal/data_loading.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--leaf", default=None, help="Substring filter on leaf name.")
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Rows per leaf; 0 reads every row.",
    )
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--json", default=None, help="Write the full report here.")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
