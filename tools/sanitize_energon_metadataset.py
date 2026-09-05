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

"""Sanitize an exported Energon MetadatasetV2 yaml so it loads.

Some ``export_subset.py`` outputs declare aux media roots as bare relative
paths::

    aux:
      media_source: media/reasoning_off__scalecua

``megatron.energon`` resolves a bare path to an ``AuxDatasetReference``, whose
``post_initialize`` asserts the target is a *prepared* Energon dataset
(``metadataset_v2.py`` -> ``AuxDatasetReference.post_initialize``)::

    AssertionError: Auxiliary datasets must be prepared Energon datasets.
    This one does not exist or is not prepared: .../media/reasoning_off__scalecua

Exported media trees are plain directories of payload files -- no ``.nv-meta``,
no shards -- so that assertion always fails. Prefixing the value with the
``filesystem://`` scheme makes energon build an ``AuxFilesystemReference``
instead, which is a plain root and needs no index
(``_normalize_aux_reference``: ``prot == "filesystem"`` branch).

Only the scheme is missing; the media itself does not need ``energon prepare``.

Checks performed:
  * every aux value gets a scheme (``filesystem://`` unless it already has one)
  * aux targets that ARE prepared datasets (have .nv-meta/index.sqlite) are left
    alone -- those legitimately want AuxDatasetReference
  * reports aux targets that do not exist on disk at all
  * reports ``path:`` entries whose .jsonl is missing

Usage:
    python tools/sanitize_energon_metadataset.py --yaml <subset.yaml> --dry-run
    python tools/sanitize_energon_metadataset.py --yaml <subset.yaml> --in-place
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Matches "<indent>media_source: <value>" and any other aux key, capturing the
# value. Deliberately line-based: these files carry comments and ordering that a
# yaml round-trip would discard, and they can be tens of thousands of lines.
_AUX_VALUE = re.compile(r"^(?P<pre>\s*)(?P<key>[A-Za-z0-9_]+):\s*(?P<val>\S+)\s*$")
_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_PREPARED = Path(".nv-meta") / "index.sqlite"


def _aux_key_lines(lines: list[str]) -> list[int]:
    """Indices of value lines that sit under an ``aux:`` block."""
    out: list[int] = []
    aux_indent: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "aux:":
            aux_indent = indent
            continue
        if aux_indent is not None:
            if indent > aux_indent:
                out.append(i)
            else:
                aux_indent = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", required=True, help="Top-level metadataset yaml (e.g. subset.yaml).")
    ap.add_argument("--scheme", default="filesystem://", help="Scheme to add. Default filesystem://")
    ap.add_argument("--in-place", action="store_true", help="Rewrite the file, keeping a .bak.")
    ap.add_argument("--dry-run", action="store_true", help="Report only.")
    a = ap.parse_args()

    y = Path(a.yaml).expanduser().resolve()
    if not y.is_file():
        raise SystemExit(f"no such file: {y}")
    root = y.parent
    lines = y.read_text().splitlines()

    changed = skipped_prepared = already = 0
    missing: list[str] = []

    for i in _aux_key_lines(lines):
        m = _AUX_VALUE.match(lines[i])
        if not m:
            continue
        val = m.group("val")
        if _URL.match(val):
            already += 1
            continue
        target = (root / val).resolve()
        if (target / _PREPARED).is_file():
            # A genuinely prepared dataset: leave it as an AuxDatasetReference.
            skipped_prepared += 1
            continue
        if not target.exists():
            missing.append(val)
        lines[i] = f"{m.group('pre')}{m.group('key')}: {a.scheme}{val}"
        changed += 1

    # Sanity: do the referenced record files exist?
    bad_paths = [
        mm.group(1)
        for ln in lines
        if (mm := re.match(r"^\s*-?\s*path:\s*(\S+)\s*$", ln))
        and not _URL.match(mm.group(1))
        and not (root / mm.group(1)).exists()
    ]

    print(f"  {y}")
    print(f"    aux values needing a scheme : {changed}")
    print(f"    already had a scheme        : {already}")
    print(f"    prepared datasets, untouched: {skipped_prepared}")
    if missing:
        print(f"    !! aux targets not on disk  : {len(missing)}")
        for p in missing[:5]:
            print(f"       {p}")
    if bad_paths:
        print(f"    !! path: entries not on disk: {len(bad_paths)}")
        for p in bad_paths[:5]:
            print(f"       {p}")

    if a.dry_run or not a.in_place:
        print("    dry run: nothing written" if a.dry_run else "    pass --in-place to write")
        return 0
    if changed:
        shutil.copy2(y, y.with_suffix(y.suffix + ".bak"))
        y.write_text("\n".join(lines) + "\n")
        print(f"    wrote {y}  (backup: {y.name}.bak)")
    else:
        print("    nothing to change")
    return 1 if (missing or bad_paths) else 0


if __name__ == "__main__":
    sys.exit(main())
