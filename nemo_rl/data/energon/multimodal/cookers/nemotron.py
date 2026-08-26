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
import logging
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from megatron.energon import (
    CachePool,
    CrudeSample,
    FileStore,
    SourceInfo,
    basic_sample_keys,
    cooker,
    stateless,
)
from megatron.energon.av import AVDecoder

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
_CONVERSATION_MEDIA_TYPES = frozenset(
    {"image", "video", "video_frame", "audio"}
)

_LOG = logging.getLogger(__name__)
_WARNED_METADATA_STORES: set[str] = set()

EXPLICIT_ASSISTANT_LOSS_COOK = "general_conversations_jsonl_explicit_loss_v1"
EXPLICIT_ASSISTANT_LOSS_MODE = "explicit_assistant_turns"
EXPLICIT_ASSISTANT_LOSS_FIELD = "conversations[*].loss"
_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "agent"})
_EXPLICIT_ASSISTANT_LOSS_INCOMPATIBLE_OPTIONS = (
    "train_only_on_last_assistant_turn",
    "skip_chat_template",
    "tool_response_as_turn_boundary",
    "offline_packed_messages",
)

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


def _metadata_values(metadata: object) -> Mapping[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, Mapping):
        return metadata
    if is_dataclass(metadata) and not isinstance(metadata, type):
        return asdict(metadata)
    try:
        return vars(metadata)
    except TypeError as error:
        raise ValueError(
            f"Unsupported Energon media metadata type {type(metadata)!r}."
        ) from error


def _media_metadata(store: FileStore, path: str) -> FrozenMediaMetadata:
    try:
        metadata = store.get_media_metadata(path)
    except (
        AttributeError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ):
        store_path = str(store.get_path())
        if store_path not in _WARNED_METADATA_STORES:
            _LOG.warning(
                "Dataset %s has no prepared media metadata for %s; "
                "the cooker will decode media to derive it.",
                store_path,
                path,
            )
            _WARNED_METADATA_STORES.add(store_path)
        return ()
    return freeze_media_metadata(_metadata_values(metadata))


def _open_media(modality: str, value: Any) -> Any:
    if modality == "image":
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            image = Image.open(BytesIO(bytes(value)))
            image.load()
            return image
        if isinstance(value, (str, Path)):
            image = Image.open(value)
            image.load()
            return image
    elif modality in {"video", "video_frame", "audio"}:
        if isinstance(value, AVDecoder):
            return value
        if isinstance(value, (str, Path)):
            value = Path(value).read_bytes()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return AVDecoder(BytesIO(bytes(value)))
    raise ValueError(f"Cannot open {modality} media from {type(value)!r}.")


def _derived_media_metadata(modality: str, value: Any) -> FrozenMediaMetadata:
    if modality == "image":
        image = _open_media(modality, value)
        return freeze_media_metadata(
            {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
            }
        )
    decoder = _open_media(modality, value)
    return freeze_media_metadata(_metadata_values(decoder.get_metadata()))


def _source_info(store: FileStore, path: str) -> SourceInfo:
    return SourceInfo(
        dataset_path=store.get_path(),
        index=path,
        shard_name=None,
        file_names=(path,),
    )


def _aux_store_and_path(
    sample: CrudeSample,
    path: str,
    *,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    strip_matched_prefix: bool,
    basename_missing_absolute: bool,
    missing_aux_source_is_error: bool,
) -> tuple[FileStore | None, str]:
    store = media_source
    if store is not None:
        media_path = path
        if (
            basename_missing_absolute
            and Path(media_path).is_absolute()
            and not Path(media_path).is_file()
        ):
            media_path = PurePosixPath(media_path).name
        return store, media_path

    clean_path = re.sub(r"(?:^\./|/\.(?=/))", "", path)
    prefixes = (sample.get("__subflavors__") or {}).get("aux_data_prefixes", {})
    for prefix, aux_key in prefixes.items():
        if clean_path.startswith(prefix):
            try:
                media_path = (
                    clean_path[len(prefix) :] if strip_matched_prefix else path
                )
                return media_sources[aux_key], media_path
            except KeyError as error:
                if not missing_aux_source_is_error:
                    return None, path
                raise ValueError(
                    f"Auxiliary media source {aux_key!r} is not available for {path!r}."
                ) from error
    return None, path


def _aux_media(
    sample: CrudeSample,
    path: str,
    *,
    modality: str,
    metadata: FrozenMediaMetadata = (),
    cache: CachePool | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    strip_matched_prefix: bool,
    basename_missing_absolute: bool,
    allow_local: bool,
    derive_missing_metadata: bool,
    missing_aux_source_is_error: bool,
) -> tuple[Any, FrozenMediaMetadata, SourceInfo | None]:
    store, media_path = _aux_store_and_path(
        sample,
        path,
        media_source=media_source,
        media_sources=media_sources,
        strip_matched_prefix=strip_matched_prefix,
        basename_missing_absolute=basename_missing_absolute,
        missing_aux_source_is_error=missing_aux_source_is_error,
    )
    if store is None:
        if not allow_local:
            raise ValueError(
                f"No configured media source matches media path {path!r}."
            )
        local_path = Path(path)
        if not local_path.is_file():
            raise ValueError(f"Cannot find media file {path!r} in configured sources.")
        if cache is None:
            raise ValueError("Local media loading requires an Energon cache pool.")
        opened = _open_media(modality, local_path)
        metadata = metadata or _derived_media_metadata(modality, opened)
        source = SourceInfo(
            dataset_path=local_path.parent,
            index=local_path.name,
            shard_name=None,
            file_names=(local_path.name,),
        )
        cache_key = f"{sample.get('__key__', '')}.{local_path.suffix.lstrip('.')}"
        return cache.to_cache(opened, cache_key), metadata, source
    if cache is None:
        raise ValueError("Auxiliary media loading requires an Energon cache pool.")
    metadata = metadata or _media_metadata(store, media_path)
    if not metadata and derive_missing_metadata:
        metadata = _derived_media_metadata(modality, cache.get(store, media_path))
    return cache.get_lazy(store, media_path), metadata, _source_info(store, media_path)


def _primary_media(
    sample: CrudeSample,
    member: str,
    *,
    modality: str,
    metadata: FrozenMediaMetadata,
    cache: CachePool | None,
    primary: FileStore | None,
) -> tuple[Any, FrozenMediaMetadata]:
    opened = _open_media(modality, sample[member])
    metadata = metadata or (
        () if primary is None else _media_metadata(primary, f".{member}")
    )
    if not metadata:
        metadata = _derived_media_metadata(modality, opened)
    if cache is None:
        return sample[member], metadata
    cache_key = f"{sample.get('__key__', '')}.{member}"
    return cache.to_cache(opened, cache_key), metadata


def _fragment_metadata(
    fragment: Mapping[str, Any], *, modality: str
) -> tuple[FrozenMediaMetadata, FrozenMediaMetadata]:
    metadata = freeze_media_metadata(fragment.get("metadata"))
    timing: dict[str, Any] = {}
    timing_keys: tuple[str, ...] = ()
    if modality == "video":
        timing_keys = ("start_time", "end_time")
    elif modality == "video_frame":
        timing_keys = ("timestamp", "frame_index", "sample_index")
    unexpected_keys = set(fragment) - {"t", "value", "metadata", *timing_keys}
    if unexpected_keys:
        raise ValueError(
            f"Unexpected {modality} fragment fields: {sorted(unexpected_keys)!r}."
        )
    for key in timing_keys:
        if key in fragment:
            timing[key] = fragment[key]
    return metadata, freeze_media_metadata(timing)


def _merge_media_metadata(
    metadata: FrozenMediaMetadata, extra: FrozenMediaMetadata
) -> FrozenMediaMetadata:
    values = dict(metadata)
    values.update(extra)
    return freeze_media_metadata(values)


def _sample_keys_with_dataset(
    sample: CrudeSample, extra_sources: tuple[SourceInfo, ...] = ()
) -> dict[str, Any]:
    sample_keys = basic_sample_keys(sample, extra_sources)
    dataset = _decode_payload(sample).get("dataset")
    if dataset is not None:
        subflavors = dict(sample_keys.get("__subflavors__", {}) or {})
        subflavors["dataset"] = dataset
        sample_keys["__subflavors__"] = subflavors
    return sample_keys


def _validate_explicit_assistant_loss(
    sample: CrudeSample, conversations: list[Any]
) -> None:
    subflavors = sample.get("__subflavors__", {}) or {}
    if subflavors.get("loss_mask_mode") != EXPLICIT_ASSISTANT_LOSS_MODE:
        raise ValueError(
            f"{EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}."
        )
    if subflavors.get("assistant_loss_mask_field") != EXPLICIT_ASSISTANT_LOSS_FIELD:
        raise ValueError(
            f"{EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"assistant_loss_mask_field={EXPLICIT_ASSISTANT_LOSS_FIELD}."
        )
    for option in _EXPLICIT_ASSISTANT_LOSS_INCOMPATIBLE_OPTIONS:
        if subflavors.get(option, False):
            raise ValueError(f"Explicit assistant loss is incompatible with {option}.")

    has_trainable_assistant = False
    for index, message in enumerate(conversations):
        if not isinstance(message, Mapping):
            raise ValueError(f"conversations[{index}] must be an object.")
        sender = message.get("from")
        if sender not in _ROLES | _ASSISTANT_ROLES | {"human"}:
            raise ValueError(
                f"conversations[{index}].from has unsupported sender {sender!r}."
            )
        if sender in _ASSISTANT_ROLES:
            if type(message.get("loss")) is not bool:
                raise ValueError(
                    f"conversations[{index}].loss must be a boolean for "
                    f"assistant sender {sender!r}."
                )
            has_trainable_assistant = has_trainable_assistant or message["loss"]
        elif "loss" in message:
            raise ValueError(
                f"conversations[{index}].loss is only valid on assistant turns."
            )
    if not has_trainable_assistant:
        raise ValueError("Explicit assistant loss requires at least one loss=true turn.")


def _validate_loss_mask_subflavors(
    sample: CrudeSample, *, explicit_assistant_loss: bool
) -> None:
    subflavors = sample.get("__subflavors__", {}) or {}
    cook_name = subflavors.get("cook")
    loss_mask_mode = subflavors.get("loss_mask_mode")
    assistant_loss_mask_field = subflavors.get("assistant_loss_mask_field")
    if loss_mask_mode not in (None, "", EXPLICIT_ASSISTANT_LOSS_MODE):
        raise ValueError(f"Unsupported loss_mask_mode={loss_mask_mode!r}.")
    if (
        "assistant_loss_mask_field" in subflavors
        and loss_mask_mode != EXPLICIT_ASSISTANT_LOSS_MODE
    ):
        raise ValueError(
            "assistant_loss_mask_field requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}."
        )

    selected_explicit_loss = loss_mask_mode == EXPLICIT_ASSISTANT_LOSS_MODE
    if selected_explicit_loss and cook_name != EXPLICIT_ASSISTANT_LOSS_COOK:
        raise ValueError(
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE} requires "
            f"cook={EXPLICIT_ASSISTANT_LOSS_COOK}."
        )
    if selected_explicit_loss and assistant_loss_mask_field != EXPLICIT_ASSISTANT_LOSS_FIELD:
        raise ValueError(
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE} requires "
            f"assistant_loss_mask_field={EXPLICIT_ASSISTANT_LOSS_FIELD}."
        )
    if cook_name == EXPLICIT_ASSISTANT_LOSS_COOK and not selected_explicit_loss:
        raise ValueError(
            f"cook={EXPLICIT_ASSISTANT_LOSS_COOK} requires "
            f"loss_mask_mode={EXPLICIT_ASSISTANT_LOSS_MODE}."
        )
    if selected_explicit_loss != explicit_assistant_loss:
        expected = EXPLICIT_ASSISTANT_LOSS_COOK if explicit_assistant_loss else "a standard cooker"
        raise ValueError(f"Loss-mask configuration does not match {expected}.")


def _apply_last_assistant_mask(
    sample: CrudeSample, messages: list[dict[str, Any]]
) -> None:
    subflavors = sample.get("__subflavors__", {}) or {}
    if not subflavors.get("train_only_on_last_assistant_turn", False):
        return
    assistant_indexes = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indexes:
        raise ValueError(
            "train_only_on_last_assistant_turn requires an assistant message."
        )
    last_assistant = assistant_indexes[-1]
    for index, message in enumerate(messages):
        message["train_on_message"] = index == last_assistant


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True)
def cook_nemotron_conversation(
    sample: CrudeSample,
    cache: CachePool | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook the Nemotron ``conversation[].fragments[]`` source schema."""
    payload = _decode_payload(sample)
    conversation = payload.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("Nemotron fragment conversations require a non-empty list.")

    messages: list[dict[str, Any]] = []
    media: list[MediaRef] = []
    source_info: list[SourceInfo] = []
    for raw_message in conversation:
        if not isinstance(raw_message, dict):
            raise ValueError("Nemotron conversation messages must be objects.")
        role = _ROLE_ALIASES.get(raw_message.get("sender"), raw_message.get("sender"))
        if role not in _ROLES:
            raise ValueError(f"Unknown Nemotron conversation role {role!r}.")
        fragments = raw_message.get("fragments")
        if not isinstance(fragments, list) or not fragments:
            raise ValueError("Nemotron conversation messages require fragments.")

        content: list[dict[str, Any]] = []
        for fragment in fragments:
            if isinstance(fragment, str):
                content.append({"type": "text", "text": fragment})
                continue
            if not isinstance(fragment, dict):
                raise ValueError("Nemotron conversation fragments must be objects.")
            fragment_type = fragment.get("t")
            value = fragment.get("value")
            if fragment_type == "text":
                if not isinstance(value, str):
                    raise ValueError("Text fragments require a string value.")
                content.append({"type": "text", "text": value})
                continue
            if fragment_type not in _CONVERSATION_MEDIA_TYPES:
                raise ValueError(
                    f"Unsupported Nemotron conversation fragment type {fragment_type!r}."
                )
            if not isinstance(value, str) or not value:
                raise ValueError("Media fragments require a non-empty path.")
            metadata, timing_metadata = _fragment_metadata(
                fragment, modality=fragment_type
            )
            media_value, store_metadata, source = _aux_media(
                sample,
                value,
                modality=fragment_type,
                metadata=metadata,
                cache=cache,
                media_source=media_source,
                media_sources=media_sources,
                strip_matched_prefix=True,
                basename_missing_absolute=False,
                allow_local=False,
                derive_missing_metadata=True,
                missing_aux_source_is_error=True,
            )
            metadata = _merge_media_metadata(
                metadata or store_metadata, timing_metadata
            )
            media_index = len(media)
            media.append(
                MediaRef(
                    modality=fragment_type,
                    value=media_value,
                    metadata=metadata,
                )
            )
            content.append({"type": fragment_type, "media_index": media_index})
            if source is not None:
                source_info.append(source)
        message = {"role": role, "content": content}
        if "loss" in raw_message:
            raise ValueError(
                "Fragment conversations do not support explicit loss fields; use "
                f"cook={EXPLICIT_ASSISTANT_LOSS_COOK}."
            )
        messages.append(message)

    _apply_last_assistant_mask(sample, messages)
    return CanonicalSFTSample(
        **_sample_keys_with_dataset(sample, tuple(source_info)),
        messages=messages,
        media=media,
        tools=None,
    )


def _resolve_media_value(
    sample: CrudeSample,
    source: Any,
    *,
    media_tag: str,
    metadata: FrozenMediaMetadata,
    lookup_source: bool,
    require_member: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    tried_default_extensions: set[str],
) -> tuple[Any, FrozenMediaMetadata, SourceInfo | None]:
    if not lookup_source or not isinstance(source, str):
        return source, (), None

    sample_key = str(sample.get("__key__", ""))
    basename = PurePosixPath(source).name
    candidates = [basename.split(".", 1)[1] if "." in basename else basename]
    for candidate in candidates:
        if require_member and candidate and candidate in sample:
            modality = "video" if media_tag == "video-audio" else media_tag
            value, metadata = _primary_media(
                sample,
                candidate,
                modality=modality,
                metadata=metadata,
                cache=cache,
                primary=primary,
            )
            return value, metadata, None
    if require_member:
        for candidate in _DEFAULT_EXTENSIONS[media_tag]:
            if candidate in tried_default_extensions or candidate not in sample:
                continue
            tried_default_extensions.add(candidate)
            modality = "video" if media_tag == "video-audio" else media_tag
            value, metadata = _primary_media(
                sample,
                candidate,
                modality=modality,
                metadata=metadata,
                cache=cache,
                primary=primary,
            )
            return value, metadata, None
    modality = "video" if media_tag == "video-audio" else media_tag
    value, metadata, source_info = _aux_media(
        sample,
        source,
        modality=modality,
        metadata=metadata,
        cache=cache,
        media_source=media_source,
        media_sources=media_sources,
        strip_matched_prefix=False,
        basename_missing_absolute=True,
        allow_local=True,
        derive_missing_metadata=True,
        missing_aux_source_is_error=False,
    )
    if source_info is not None:
        return value, metadata, source_info
    if require_member:
        raise ValueError(
            f"Media member {source!r} is absent from Energon sample {sample_key!r}."
        )
    return source, (), None


def _normalized_media_fields(payload: dict[str, Any]) -> dict[str, list[Any]]:
    fields: dict[str, list[Any]] = {}
    aliases = {
        "audio": ("sound", "speech", "speeches", "audio", "audios"),
        "image": ("image", "images"),
        "video": ("video", "videos"),
        "video-audio": ("video-sound",),
    }
    for canonical, source_fields in aliases.items():
        for field in source_fields:
            if field in payload:
                value = payload[field]
                fields[canonical] = value if isinstance(value, list) else [value]
    return fields


def _append_media_part(
    *,
    sample: CrudeSample,
    entry: Any,
    media_tag: str,
    require_member: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
    media: list[MediaRef],
    content: list[dict[str, Any]],
    source_info: list[SourceInfo],
    tried_default_extensions: set[str],
) -> None:
    source, metadata, lookup_source = _descriptor(entry)
    value, store_metadata, resolved_source = _resolve_media_value(
        sample,
        source,
        media_tag=media_tag,
        metadata=metadata,
        lookup_source=lookup_source,
        require_member=require_member,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
        tried_default_extensions=tried_default_extensions,
    )
    metadata = metadata or store_metadata
    modalities = ("video", "audio") if media_tag == "video-audio" else (media_tag,)
    for modality in modalities:
        if resolved_source is not None:
            source_info.append(resolved_source)
        media_index = len(media)
        media.append(MediaRef(modality=modality, value=value, metadata=metadata))
        content.append({"type": modality, "media_index": media_index})


def _cook_general_payload(
    sample: CrudeSample,
    payload: dict[str, Any],
    *,
    require_member: bool,
    explicit_assistant_loss: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
) -> CanonicalSFTSample:
    conversations = payload.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Nemotron samples require a non-empty conversations list.")
    _validate_loss_mask_subflavors(
        sample, explicit_assistant_loss=explicit_assistant_loss
    )
    if explicit_assistant_loss:
        _validate_explicit_assistant_loss(sample, conversations)

    media_fields = _normalized_media_fields(payload)
    media_indexes: defaultdict[str, int] = defaultdict(int)
    messages: list[dict[str, Any]] = []
    media: list[MediaRef] = []
    source_info: list[SourceInfo] = []
    tried_default_extensions: set[str] = set()

    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError("Each Nemotron conversation turn must be an object.")
        if "loss" in message and not explicit_assistant_loss:
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
                    cache=cache,
                    primary=primary,
                    media_source=media_source,
                    media_sources=media_sources,
                    media=media,
                    content=content,
                    source_info=source_info,
                    tried_default_extensions=tried_default_extensions,
                )
                media_indexes[media_tag] += 1
            elif part.strip():
                content.append({"type": "text", "text": part})
                has_text = True
        if not has_text:
            content.append({"type": "text", "text": " "})
        cooked_message = {"role": role, "content": content}
        if explicit_assistant_loss and role == "assistant":
            cooked_message["train_on_message"] = message["loss"]
        messages.append(cooked_message)

    for media_tag, used in media_indexes.items():
        entries = media_fields[media_tag]
        if used != len(entries):
            raise ValueError(
                f"Retrieved {used}/{len(entries)} {media_tag} media values from sample "
                f"{sample.get('__key__', '')!r}."
            )

    _apply_last_assistant_mask(sample, messages)
    return CanonicalSFTSample(
        **_sample_keys_with_dataset(sample, tuple(source_info)),
        messages=messages,
        media=media,
        tools=None,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_webdataset(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one monolithic Nemotron conversation without opening its media."""
    return _cook_general_payload(
        sample,
        _decode_payload(sample),
        require_member=True,
        explicit_assistant_loss=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_jsonl(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one Nemotron JSONL conversation with lazy media paths."""
    return _cook_general_payload(
        sample,
        _decode_payload(sample),
        require_member=False,
        explicit_assistant_loss=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_general_conversations_jsonl_explicit_loss_v1(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook versioned JSONL with explicit per-assistant-turn loss flags."""
    return _cook_general_payload(
        sample,
        _decode_payload(sample),
        require_member=False,
        explicit_assistant_loss=True,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


def _cook_granary(
    sample: CrudeSample,
    *,
    require_member: bool,
    cache: CachePool | None,
    primary: FileStore | None,
    media_source: FileStore | None,
    media_sources: dict[str, FileStore],
) -> CanonicalSFTSample:
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
    return _cook_general_payload(
        sample,
        granary_payload,
        require_member=require_member,
        explicit_assistant_loss=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_granary_english_webdataset(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one Granary WDS row without opening its audio member."""
    return _cook_granary(
        sample,
        require_member=True,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


@supports_model_families("nemotron")
@stateless
@cooker(need_cache=True, need_primary=True)
def cook_granary_english_jsonl(
    sample: CrudeSample,
    cache: CachePool | None = None,
    primary: FileStore | None = None,
    media_source: FileStore | None = None,
    **media_sources: FileStore,
) -> CanonicalSFTSample:
    """Cook one Granary JSONL row with a lazy audio path."""
    return _cook_granary(
        sample,
        require_member=False,
        cache=cache,
        primary=primary,
        media_source=media_source,
        media_sources=media_sources,
    )


__all__ = [
    "GRANARY_ENGLISH_PROMPT",
    "cook_general_conversations_jsonl",
    "cook_general_conversations_jsonl_explicit_loss_v1",
    "cook_general_conversations_webdataset",
    "cook_granary_english_jsonl",
    "cook_granary_english_webdataset",
    "cook_nemotron_conversation",
]
