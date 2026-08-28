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

"""Backfill inline media metadata into an exported Energon subset.

An exported subset carries media metadata inline on each conversation
fragment rather than in ``.nv-meta`` sidecars::

    {"t": "image", "value": "images/000417.jpg",
     "metadata": {"width": 371, "height": 100, "format": "JPEG", "mode": "RGB"}}

Leaves exported without it cost both stacks:

* NeMo-RL survives, because ``cookers/nemotron.py:301`` falls back to
  ``_derived_media_metadata`` and decodes the file, but it pays that decode on
  every row -- the "has no prepared media metadata; the cooker will decode
  media to derive it" warnings. Filling the metadata removes the decode from
  pre-encode entirely.
* The Megatron reference CRASHES on image rows. ``ImageMedia.width`` is
  ``return self.metadata["width"]`` (``conversation_sample.py:29``) with
  ``metadata`` defaulting to ``None``, reached from
  ``image_processing.py:769``. Measured on cook_subset_v2: 54 of 66 reference
  rejections, eliminated outright by this backfill.

Both media types are handled, using the exact key set the already-populated
leaves use, so a backfilled fragment is indistinguishable from an original:

    image   width, height, format, mode                       (PIL, header only)
    video   video_duration, video_num_frames, video_fps,      (PyAV, header only)
            video_width, video_height, audio_duration,
            audio_channels, audio_sample_rate

Rewriting a jsonl invalidates its ``.idx`` sidecar -- a table of
``(N + 1)`` little-endian uint64 line offsets -- so that is regenerated too.
Skipping it leaves every offset past the first modified row pointing into the
middle of a line.

Usage:
    # see what would change
    uv run --locked --extra energon tools/backfill_media_metadata.py \
        --subset ~/data/super-test-blend/cook_subset_v2 --dry-run

    # write alongside the originals as <leaf>.withmeta.jsonl
    uv run --locked --extra energon tools/backfill_media_metadata.py \
        --subset ~/data/super-test-blend/cook_subset_v2

    # overwrite in place, keeping a .bak
    uv run --locked --extra energon tools/backfill_media_metadata.py \
        --subset ~/data/super-test-blend/cook_subset_v2 --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any, Callable

# The fields that decide whether a fragment needs filling, per media type. The
# reference reads image dimensions through metadata["width"]; video goes
# through VideoFrameMedia.video_width, which is why only images crash.
_REQUIRED: dict[str, tuple[str, ...]] = {
    "image": ("width", "height"),
    "video": ("video_width", "video_height", "video_num_frames"),
}


def _image_metadata(path: Path) -> dict[str, Any]:
    """Read image dimensions without decoding pixels.

    PIL.Image.open parses the header lazily, so this stays fast over thousands
    of files.
    """
    from PIL import Image

    with Image.open(path) as image:
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
        }


def _video_metadata(path: Path) -> dict[str, Any]:
    """Read video and audio properties from the container header.

    Verified to reproduce the recorded metadata of an already-populated leaf
    field for field, including the float durations.
    """
    import av

    with av.open(str(path)) as container:
        video = container.streams.video[0]
        metadata: dict[str, Any] = {
            "video_duration": (
                float(video.duration * video.time_base)
                if video.duration
                else float(container.duration / av.time_base)
            ),
            "video_num_frames": video.frames,
            "video_fps": float(video.average_rate),
            "video_width": video.codec_context.width,
            "video_height": video.codec_context.height,
        }
        # Always emit the audio keys. A silent video records them as explicit
        # nulls, not as absent keys, so omitting them would turn a downstream
        # metadata["audio_duration"] returning None into a KeyError.
        audio = container.streams.audio[0] if container.streams.audio else None
        metadata["audio_duration"] = (
            float(audio.duration * audio.time_base)
            if audio is not None and audio.duration
            else None
        )
        metadata["audio_channels"] = None if audio is None else audio.channels
        metadata["audio_sample_rate"] = None if audio is None else audio.sample_rate
        return metadata


_READERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "image": _image_metadata,
    "video": _video_metadata,
}


def _needs_metadata(fragment: Any) -> bool:
    """True when this fragment is media and lacks its required fields."""
    if not isinstance(fragment, dict):
        return False
    required = _REQUIRED.get(fragment.get("t"))
    if required is None:
        return False
    metadata = fragment.get("metadata") or {}
    return any(metadata.get(field) is None for field in required)


def _write_index(jsonl: Path) -> None:
    """Rebuild the .idx offset sidecar for a rewritten jsonl.

    Format, verified byte for byte against untouched originals: ``(N + 1)``
    little-endian uint64 offsets -- the start of every line, then EOF.
    """
    offsets = [0]
    with jsonl.open("rb") as handle:
        for line in handle:
            offsets.append(offsets[-1] + len(line))
    index = jsonl.with_suffix(jsonl.suffix + ".idx")
    temporary = index.with_suffix(index.suffix + ".tmp")
    temporary.write_bytes(struct.pack(f"<{len(offsets)}Q", *offsets))
    temporary.replace(index)


def process_leaf(
    jsonl: Path, media_root: Path, *, in_place: bool, dry_run: bool
) -> dict[str, int]:
    """Fill missing media metadata for one leaf, and refresh its index."""
    stats = dict.fromkeys(
        ("rows", "missing", "filled", "unreadable", "absent"), 0
    )
    cache: dict[Path, dict[str, Any] | None] = {}
    lines: list[str] = []

    with jsonl.open() as handle:
        for line in handle:
            stats["rows"] += 1
            try:
                row = json.loads(line)
            except ValueError:
                lines.append(line)
                continue
            for message in row.get("conversation", []) or []:
                for fragment in message.get("fragments", []) or []:
                    if not _needs_metadata(fragment):
                        continue
                    stats["missing"] += 1
                    path = media_root / str(fragment.get("value", ""))
                    if path not in cache:
                        cache[path] = _read(path, fragment["t"], stats)
                    metadata = cache[path]
                    if metadata is None:
                        continue
                    # Merge, so unrelated keys already on the fragment survive.
                    fragment["metadata"] = {
                        **(fragment.get("metadata") or {}),
                        **metadata,
                    }
                    stats["filled"] += 1
            lines.append(json.dumps(row) + "\n")

    if dry_run or not stats["missing"]:
        return stats

    target = jsonl if in_place else jsonl.with_suffix(".withmeta.jsonl")
    if in_place:
        backup = jsonl.with_suffix(jsonl.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(jsonl, backup)
    # Temp file then rename: an interrupted run must never leave a truncated
    # dataset where the original was.
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text("".join(lines))
    temporary.replace(target)
    _write_index(target)
    return stats


def _read(path: Path, modality: str, stats: dict[str, int]) -> dict[str, Any] | None:
    """Read one media file's metadata, counting failures rather than raising."""
    if not path.exists():
        stats["absent"] += 1
        return None
    try:
        return _READERS[modality](path)
    except Exception as error:  # noqa: BLE001 - one bad file must not abort the run
        print(f"    cannot read {path}: {type(error).__name__}: {error}")
        stats["unreadable"] += 1
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, help="Subset root directory.")
    parser.add_argument(
        "--media-root",
        default=None,
        help=(
            "Media directory. Default <subset>/media, with each leaf's files "
            "under a subdirectory named after the leaf."
        ),
    )
    parser.add_argument(
        "--leaf",
        action="append",
        default=[],
        help="Leaf name without .jsonl, repeatable. Default: every leaf.",
    )
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.subset).expanduser().resolve()
    datasets = root / "datasets"
    if not datasets.is_dir():
        raise SystemExit(f"no datasets/ directory under {root}")
    media = Path(args.media_root).expanduser() if args.media_root else root / "media"

    leaves = (
        [datasets / f"{name}.jsonl" for name in args.leaf]
        if args.leaf
        else sorted(datasets.glob("*.jsonl"))
    )

    totals = dict.fromkeys(("missing", "filled", "unreadable", "absent"), 0)
    touched = 0
    for jsonl in leaves:
        if not jsonl.exists():
            print(f"{jsonl.name}: NOT FOUND")
            continue
        stats = process_leaf(
            jsonl, media / jsonl.stem, in_place=args.in_place, dry_run=args.dry_run
        )
        if stats["missing"]:
            touched += 1
            print(
                f"{jsonl.stem:56s} rows={stats['rows']:5d} "
                f"missing={stats['missing']:5d} filled={stats['filled']:5d} "
                f"unreadable={stats['unreadable']:3d} absent={stats['absent']:3d}"
            )
        for key in totals:
            totals[key] += stats[key]

    print(
        f"\n{touched} leaves needed metadata: {totals['filled']} filled of "
        f"{totals['missing']} ({totals['unreadable']} unreadable, "
        f"{totals['absent']} absent)"
    )
    if args.dry_run:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
