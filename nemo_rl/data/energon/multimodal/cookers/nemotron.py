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

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

from megatron.energon import CrudeSample, basic_sample_keys, stateless

from nemo_rl.data.energon.multimodal.model_families import supports_model_families
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    FrozenMediaMetadata,
    MediaRef,
    freeze_media_metadata,
)

# Source parsing follows the Apache-2.0 NeMo-RL implementation in
# nemo_rl/data/datasets/response_datasets/general_conversations_dataset.py.
# Tag names, extension fallback, Granary prompt text, and video/audio ordering
# were checked against Megatron-LM 6822175d92a40e0528be905aee50f5930cfa0c98.
_FIELD_ALIASES = {
    "speech": "audio",
    "speeches": "audio",
    "sound": "audio",
    "audios": "audio",
    "images": "image",
    "videos": "video",
    "video-sound": "video-audio",
}
_TAG_ALIASES = {
    **_FIELD_ALIASES,
    "image": "image",
    "video": "video",
    "audio": "audio",
    "video-audio": "video-audio",
}
_TAG_PATTERN = re.compile(
    "("
    + "|".join(
        f"<{re.escape(tag)}>" for tag in sorted(_TAG_ALIASES, key=len, reverse=True)
    )
    + ")"
)
_DEFAULT_EXTENSIONS = {
    "image": ("png", "jpeg", "jpg", "img"),
    "video": ("mp4",),
    "video-audio": ("mp4",),
    "audio": ("wav", "flac", "mp3"),
}
_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "agent": "assistant"}
_ROLES = frozenset({"system", "user", "assistant", "tool"})

GRANARY_ENGLISH_PROMPT = (
    "<audio>. \nTranscribe the spoken content to written english text, "
    "with punctuations and capitalizations."
)


def _decode_payload(sample: CrudeSample) -> dict[str, Any]:
    value: Any = sample.get("json", sample)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Nemotron conversation data must decode to an object.")
    return dict(value)


def _descriptor(entry: Any) -> tuple[Any, FrozenMediaMetadata, bool]:
    """Return the media source, frozen metadata, and whether source lookup applies."""
    if not isinstance(entry, dict):
        return entry, (), True

    metadata = freeze_media_metadata(entry.get("metadata"))
    if "value" in entry:
        return entry["value"], metadata, False
    for key in ("member", "path"):
        if key in entry:
            return entry[key], metadata, True
    raise ValueError("Media entries need a value, member, or path.")


def _resolve_media_value(
    sample: CrudeSample,
    source: Any,
    *,
    media_tag: str,
    lookup_source: bool,
    require_member: bool,
) -> Any:
    if not lookup_source or not isinstance(source, str):
        return source

    source_path = PurePosixPath(source)
    sample_key = str(sample.get("__key__", ""))
    candidates = [source, source_path.name]
    if "." in source_path.name:
        candidates.append(source_path.name.split(".", 1)[1])
    if source_path.suffix:
        candidates.append(source_path.suffix.removeprefix("."))
    if sample_key and source.startswith(f"{sample_key}."):
        candidates.append(source.removeprefix(f"{sample_key}."))
    candidates.extend(_DEFAULT_EXTENSIONS[media_tag])

    for candidate in candidates:
        if candidate and candidate in sample:
            return sample[candidate]
    if require_member:
        raise ValueError(
            f"Media member {source!r} is absent from Energon sample {sample_key!r}."
        )
    return source


def _normalized_media_fields(payload: dict[str, Any]) -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = {}
    for field in _TAG_ALIASES:
        if field not in payload:
            continue
        canonical = _TAG_ALIASES[field]
        if canonical in fields:
            raise ValueError(
                f"Nemotron media fields {field!r} and another alias both map to {canonical!r}."
            )
        value = payload[field]
        fields[canonical] = value if isinstance(value, list) else [value]
    return fields


def _append_media_part(
    *,
    sample: CrudeSample,
    entry: Any,
    media_tag: str,
    require_member: bool,
    media: list[MediaRef],
    content: list[dict[str, Any]],
) -> None:
    source, metadata, lookup_source = _descriptor(entry)
    value = _resolve_media_value(
        sample,
        source,
        media_tag=media_tag,
        lookup_source=lookup_source,
        require_member=require_member,
    )
    modalities = ("video", "audio") if media_tag == "video-audio" else (media_tag,)
    for modality in modalities:
        media_index = len(media)
        media.append(MediaRef(modality=modality, value=value, metadata=metadata))
        content.append({"type": modality, "media_index": media_index})


def _cook_general_payload(
    sample: CrudeSample,
    payload: dict[str, Any],
    *,
    require_member: bool,
) -> CanonicalSFTSample:
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Nemotron samples require a non-empty conversations list.")

    media_fields = _normalized_media_fields(payload)
    media_indexes: defaultdict[str, int] = defaultdict(int)
    messages: list[dict[str, Any]] = []
    media: list[MediaRef] = []

    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("Each Nemotron conversation turn must be an object.")
        if "loss" in message:
            raise ValueError(
                "Standard Nemotron conversation cookers do not accept explicit loss fields."
            )
        role = _ROLE_ALIASES.get(message.get("from"), message.get("from"))
        if role not in _ROLES:
            raise ValueError(
                f"Unknown Nemotron conversation role {message.get('from')!r}."
            )
        value = message.get("value")
        if not isinstance(value, str):
            raise ValueError("Each Nemotron conversation value must be a string.")

        content: list[dict[str, Any]] = []
        has_text = False
        for part in re.split(_TAG_PATTERN, value):
            tag_name = (
                part[1:-1] if part.startswith("<") and part.endswith(">") else None
            )
            if tag_name in _TAG_ALIASES:
                media_tag = _TAG_ALIASES[tag_name]
                entries = media_fields.get(media_tag)
                media_index = media_indexes[media_tag]
                if entries is None:
                    raise ValueError(
                        f"Tag <{tag_name}> has no {media_tag!r} media field."
                    )
                if media_index >= len(entries):
                    raise ValueError(f"Tag <{tag_name}> has no remaining media value.")
                _append_media_part(
                    sample=sample,
                    entry=entries[media_index],
                    media_tag=media_tag,
                    require_member=require_member,
                    media=media,
                    content=content,
                )
                media_indexes[media_tag] += 1
            elif part.strip():
                content.append({"type": "text", "text": part})
                has_text = True
        if not has_text:
            content.append({"type": "text", "text": " "})
        messages.append({"role": role, "content": content})

    for media_tag, entries in media_fields.items():
        used = media_indexes[media_tag]
        if used != len(entries):
            raise ValueError(
                f"Retrieved {used}/{len(entries)} {media_tag} media values from sample "
                f"{sample.get('__key__', '')!r}."
            )

    return CanonicalSFTSample(
        **basic_sample_keys(sample),
        messages=messages,
        media=media,
        tools=None,
    )


@supports_model_families("nemotron")
@stateless
def cook_general_conversations_webdataset(sample: CrudeSample) -> CanonicalSFTSample:
    """Cook one monolithic Nemotron conversation without opening its media."""
    return _cook_general_payload(sample, _decode_payload(sample), require_member=True)


@supports_model_families("nemotron")
@stateless
def cook_general_conversations_jsonl(sample: CrudeSample) -> CanonicalSFTSample:
    """Cook one Nemotron JSONL conversation with lazy media paths."""
    return _cook_general_payload(sample, _decode_payload(sample), require_member=False)


def _cook_granary(sample: CrudeSample, *, require_member: bool) -> CanonicalSFTSample:
    payload = _decode_payload(sample)
    if "audio_filepath" not in payload:
        raise ValueError("Granary samples require audio_filepath.")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("Granary samples require a string text field.")
    granary_payload = {
        "audio": payload["audio_filepath"],
        "conversations": [
            {"from": "user", "value": GRANARY_ENGLISH_PROMPT},
            {"from": "assistant", "value": text},
        ],
    }
    return _cook_general_payload(sample, granary_payload, require_member=require_member)


@supports_model_families("nemotron")
@stateless
def cook_granary_english_webdataset(sample: CrudeSample) -> CanonicalSFTSample:
    """Cook one Granary WDS row without opening its audio member."""
    return _cook_granary(sample, require_member=True)


@supports_model_families("nemotron")
@stateless
def cook_granary_english_jsonl(sample: CrudeSample) -> CanonicalSFTSample:
    """Cook one Granary JSONL row with a lazy audio path."""
    return _cook_granary(sample, require_member=False)


__all__ = [
    "GRANARY_ENGLISH_PROMPT",
    "cook_general_conversations_jsonl",
    "cook_general_conversations_webdataset",
    "cook_granary_english_jsonl",
    "cook_granary_english_webdataset",
]
