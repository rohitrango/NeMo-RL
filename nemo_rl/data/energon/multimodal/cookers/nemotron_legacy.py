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

"""Nemotron Nano text and legacy audio source cookers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from megatron.energon import CrudeSample, basic_sample_keys, stateless

from nemo_rl.data.energon.multimodal.model_families import supports_model_families
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    FrozenMediaMetadata,
    MediaRef,
    freeze_media_metadata,
)

# Nano cleanup and the legacy source schemas follow the NVIDIA BSD-3-Clause
# Megatron-LM examples pinned at 6822175d92a40e0528be905aee50f5930cfa0c98:
# examples/multimodal/data_loading/cookers/{conversation,audio_conversation,
# omcat_legacy_audio_conversation}.py.
NO_TOOL_SYSTEM_CONTENT = (
    "<|im_start|>system\n"
    "You are a helpful and harmless assistant.\n\n"
    "You are not allowed to use any tools.<|im_end|>\n"
)
LEGACY_SYSTEM_CONTENT = (
    "<|im_start|>system\nYou are a helpful and harmless assistant.<|im_end|>\n"
)
EMPTY_SYSTEM_CONTENT = "<|im_start|>system\n<|im_end|>\n"

_NANO_ROLE_ALIASES = {"human": "user", "gpt": "assistant", "function": "tool"}
_NANO_ROLES = frozenset({"system", "user", "assistant", "tool"})
_AUDIO_TAG_PATTERN = re.compile(r"<(image|video|sound|video-sound)>")
_OMCAT_TAG_ALIASES = {
    "speech": "sound",
    "speeches": "sound",
    "audio": "sound",
    "audios": "sound",
    "images": "image",
    "videos": "video",
}
_OMCAT_TAG_PATTERN = re.compile(
    r"<(image|video|sound|video-sound|speech|speeches|audio|audios|images|videos)>"
)
_OMCAT_MEMBER_EXTENSIONS = {
    "image": frozenset({"png", "jpeg", "jpg", "img"}),
    "video": frozenset({"mp4"}),
    "sound": frozenset({"wav", "flac", "mp3"}),
}


def _decode_payload(sample: CrudeSample) -> dict[str, Any]:
    payload: Any = sample.get("json", sample)
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("Nemotron cooker input must decode to an object.")
    return dict(payload)


def _canonical_sample(
    sample: CrudeSample,
    *,
    messages: list[dict[str, Any]],
    media: list[MediaRef] | None = None,
    offline_packed_messages: bool = False,
) -> CanonicalSFTSample:
    sample_keys = basic_sample_keys(sample)
    if offline_packed_messages:
        subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
        subflavors["offline_packed_messages"] = True
        sample_keys["__subflavors__"] = subflavors
    return CanonicalSFTSample(
        **sample_keys,
        messages=messages,
        media=[] if media is None else media,
        tools=None,
    )


def _nano_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Nano message content must be text or a list of text parts.")

    fragments: list[str] = []
    for part in content:
        if isinstance(part, str):
            fragments.append(part)
            continue
        if not isinstance(part, Mapping):
            raise ValueError("Nano message content parts must be strings or objects.")
        part_type = part.get("type", part.get("t"))
        if part_type not in (None, "text"):
            raise ValueError(
                f"Nano text cookers do not support content type {part_type!r}."
            )
        text = part.get("text") or part.get("content") or part.get("value") or ""
        if not isinstance(text, str):
            raise ValueError("Nano text content values must be strings.")
        fragments.append(text)
    return "".join(fragments)


def _normalize_nano_messages(
    raw_messages: list[object],
) -> list[dict[str, Any]]:
    messages: list[dict[str, str]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            raise ValueError("Nano messages entries must be objects.")
        if "loss" in raw_message:
            raise ValueError("Nano text cookers do not accept explicit loss fields.")
        role = raw_message.get("role")
        if not isinstance(role, str):
            raise ValueError("Nano messages require a string role.")
        role = _NANO_ROLE_ALIASES.get(role, role)
        if role not in _NANO_ROLES:
            raise ValueError(f"Unsupported Nano message role {role!r}.")
        messages.append(
            {"role": role, "text": _nano_content(raw_message.get("content"))}
        )

    if not messages:
        raise ValueError("Nano text cookers require a non-empty messages list.")
    if messages[0]["role"] != "system":
        if messages[0]["text"].startswith(EMPTY_SYSTEM_CONTENT):
            messages[0]["text"] = messages[0]["text"].replace(EMPTY_SYSTEM_CONTENT, "")
        messages.insert(0, {"role": "system", "text": EMPTY_SYSTEM_CONTENT})
    elif messages[0]["text"] in (NO_TOOL_SYSTEM_CONTENT, LEGACY_SYSTEM_CONTENT):
        messages[0]["text"] = EMPTY_SYSTEM_CONTENT

    empty_think = "<|im_end|>\n<|im_start|>assistant\n<think></think>\n"
    for message in messages:
        if message["role"] == "tool":
            message["role"] = "user"
        if message["role"] == "user" and empty_think in message["text"]:
            message["text"] = message["text"].replace(
                empty_think, empty_think.removesuffix("\n")
            )
        elif message["role"] == "assistant":
            message["text"] = message["text"].rstrip() + "\n"

    open_think = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    for index, message in enumerate(messages):
        if message["role"] == "user" and index < len(messages) - 1:
            next_message = messages[index + 1]
            if message["text"].endswith(open_think) and next_message["text"].startswith(
                "\n</think>"
            ):
                message["text"] = message["text"].removesuffix(open_think) + (
                    "<|im_end|>\n<|im_start|>assistant\n<think></think>"
                )
                next_message["text"] = next_message["text"][
                    len("\n</think>") :
                ].lstrip()
        elif (
            message["role"] == "assistant"
            and index > 0
            and message["text"].startswith("\n")
            and messages[index - 1]["text"].endswith("\n")
        ):
            message["text"] = message["text"].lstrip()

    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["text"]}],
        }
        for message in messages
    ]


def _raw_nano_messages(sample: CrudeSample) -> list[object]:
    messages = _decode_payload(sample).get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Nano text cookers require a non-empty messages list.")
    return messages


@supports_model_families("nemotron")
@stateless
def cook_nano_openai_messages_jsonl(sample: CrudeSample) -> CanonicalSFTSample:
    """Normalize one Nano OpenAI-style text conversation."""
    messages = _normalize_nano_messages(_raw_nano_messages(sample))
    if any(message["role"] == "system" for message in messages[1:]):
        raise ValueError("Nano JSONL supports only one leading system message.")
    return _canonical_sample(sample, messages=messages)


def _split_nano_messages(raw_messages: list[object]) -> list[list[object]]:
    conversations: list[list[object]] = []
    current: list[object] = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            raise ValueError("Nano messages entries must be objects.")
        if message.get("role") == "system" and current:
            conversations.append(current)
            current = []
        current.append(message)
    if current:
        conversations.append(current)
    return conversations


@supports_model_families("nemotron")
@stateless
def cook_nano_openai_messages_offline_packed_jsonl(
    sample: CrudeSample,
) -> CanonicalSFTSample:
    """Normalize a Nano row that contains several pre-packed conversations."""
    messages = [
        message
        for conversation in _split_nano_messages(_raw_nano_messages(sample))
        for message in _normalize_nano_messages(conversation)
    ]
    return _canonical_sample(
        sample,
        messages=messages,
        offline_packed_messages=True,
    )


def _media_descriptor(entry: Any) -> tuple[Any, FrozenMediaMetadata]:
    if not isinstance(entry, Mapping):
        return entry, ()
    metadata = freeze_media_metadata(entry.get("metadata"))
    for key in ("value", "path", "member"):
        if key in entry:
            return entry[key], metadata
    raise ValueError("Legacy audio media entries need a value, path, or member.")


def _media_parts(
    *,
    payload: Mapping[str, Any],
    tag: str,
    media: list[MediaRef],
) -> list[dict[str, Any]]:
    prefixes = ("vis_video", "vis_sound") if tag == "video-sound" else (tag,)
    parts: list[dict[str, Any]] = []
    for prefix in prefixes:
        matches = [entry for key, entry in payload.items() if key.startswith(prefix)]
        if not matches:
            raise ValueError(
                f"Tag <{tag}> has no legacy field starting with {prefix!r}."
            )
        modality = "video" if prefix in ("video", "vis_video") else prefix
        modality = "audio" if modality in ("sound", "vis_sound") else modality
        for entry in matches:
            value, metadata = _media_descriptor(entry)
            media_index = len(media)
            media.append(MediaRef(modality=modality, value=value, metadata=metadata))
            parts.append({"type": modality, "media_index": media_index})
    return parts


def _legacy_message(
    *,
    raw_message: object,
    tag_pattern: re.Pattern[str],
    normalize_tag: dict[str, str],
    resolve_tag: Callable[[str], list[dict[str, Any]]],
    seen_tags: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_message, Mapping):
        raise ValueError("Legacy audio conversation turns must be objects.")
    sender = raw_message.get("from")
    if not isinstance(sender, str):
        raise ValueError("Legacy audio conversation turns require a string sender.")
    role = {"human": "user", "gpt": "assistant"}.get(sender)
    if role is None:
        raise ValueError(f"Unknown legacy audio sender {sender!r}.")
    value = raw_message.get("value")
    if not isinstance(value, str):
        raise ValueError("Legacy audio conversation values must be strings.")

    content: list[dict[str, Any]] = []
    for index, part in enumerate(re.split(tag_pattern, value)):
        if index % 2 == 1:
            canonical_tag = normalize_tag.get(part, part)
            if canonical_tag in seen_tags:
                raise ValueError(f"Tag <{canonical_tag}> appears more than once.")
            seen_tags.add(canonical_tag)
            content.extend(resolve_tag(canonical_tag))
        elif part.strip():
            content.append({"type": "text", "text": part})
    return {"role": role, "content": content}


@supports_model_families("nemotron")
@stateless
def cook_audio_conversation_jsonl(sample: CrudeSample) -> CanonicalSFTSample:
    """Cook the polylithic Nemotron audio-conversation source schema."""
    payload = _decode_payload(sample)
    raw_messages = payload.get("conversations")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Audio conversations require a non-empty conversations list.")
    media: list[MediaRef] = []
    seen_tags: set[str] = set()
    messages = [
        _legacy_message(
            raw_message=message,
            tag_pattern=_AUDIO_TAG_PATTERN,
            normalize_tag={},
            resolve_tag=lambda tag: _media_parts(payload=payload, tag=tag, media=media),
            seen_tags=seen_tags,
        )
        for message in raw_messages
    ]
    for tag in ("image", "video", "sound", "video-sound"):
        if tag in payload and tag not in seen_tags:
            raise ValueError(
                f"Legacy media field {tag!r} is not used by a message tag."
            )
    return _canonical_sample(sample, messages=messages, media=media)


def _omcat_member(sample: CrudeSample, tag: str) -> Any:
    if tag == "video-sound":
        raise ValueError(
            "OMCAT video-sound has no defined member mapping in the pinned reference."
        )
    extensions = _OMCAT_MEMBER_EXTENSIONS[tag]
    matches = [key for key in sample if key.lower() in extensions]
    if len(matches) != 1:
        raise ValueError(
            f"OMCAT tag <{tag}> needs exactly one member with extension "
            f"{sorted(extensions)!r}; found {matches!r}."
        )
    return sample[matches[0]]


@supports_model_families("nemotron")
@stateless
def cook_omcat_legacy_conversation_monolithic(
    sample: CrudeSample,
) -> CanonicalSFTSample:
    """Cook the OMCAT monolithic extension-keyed source schema."""
    payload = _decode_payload(sample)
    for alias, canonical in _OMCAT_TAG_ALIASES.items():
        if alias not in payload:
            continue
        if canonical in payload:
            raise ValueError(
                f"OMCAT fields {alias!r} and {canonical!r} cannot both be present."
            )
        payload[canonical] = payload.pop(alias)

    raw_messages = payload.get("conversations")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("OMCAT samples require a non-empty conversations list.")
    media: list[MediaRef] = []
    seen_tags: set[str] = set()

    def resolve_tag(tag: str) -> list[dict[str, Any]]:
        media_index = len(media)
        modality = "audio" if tag == "sound" else tag
        media.append(MediaRef(modality=modality, value=_omcat_member(sample, tag)))
        return [{"type": modality, "media_index": media_index}]

    messages = [
        _legacy_message(
            raw_message=message,
            tag_pattern=_OMCAT_TAG_PATTERN,
            normalize_tag=_OMCAT_TAG_ALIASES,
            resolve_tag=resolve_tag,
            seen_tags=seen_tags,
        )
        for message in raw_messages
    ]
    for tag in ("image", "video", "sound", "video-sound"):
        if tag in payload and tag not in seen_tags:
            raise ValueError(f"OMCAT media field {tag!r} is not used by a message tag.")
    return _canonical_sample(sample, messages=messages, media=media)


__all__ = [
    "EMPTY_SYSTEM_CONTENT",
    "LEGACY_SYSTEM_CONTENT",
    "NO_TOOL_SYSTEM_CONTENT",
    "cook_audio_conversation_jsonl",
    "cook_nano_openai_messages_jsonl",
    "cook_nano_openai_messages_offline_packed_jsonl",
    "cook_omcat_legacy_conversation_monolithic",
]
