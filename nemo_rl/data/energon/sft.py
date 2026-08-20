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

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

import torch
from megatron.energon import (
    Cooker,
    CrudeSample,
    DefaultTaskEncoder,
    Sample,
    SampleDecoder,
    basic_sample_keys,
    edataclass,
    stateless,
)

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_MEDIA_TYPES = frozenset({"image", "video", "audio"})


@dataclass(frozen=True)
class MediaRef:
    """One ordered media occurrence in a conversation."""

    modality: str
    value: Any


@edataclass
class CanonicalSFTSample(Sample):
    """Model-neutral conversation produced by the Energon cooker."""

    messages: list[dict[str, Any]]
    media: list[MediaRef]
    tools: list[dict[str, Any]] | None


@edataclass
class EncodedSFTSample(Sample):
    """One tokenized message log before batching."""

    message_log: list[dict[str, Any]]
    length: int
    loss_multiplier: float
    group_key: tuple[Any, ...]
    sample_key: str


class SFTProcessorAdapter(Protocol):
    """Temporary v1 boundary between canonical and model-specific SFT data."""

    @property
    def fingerprint(self) -> str: ...

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample: ...


def _decode_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("The SFT payload must decode to a JSON object.")


def _get_crude_payload(sample: CrudeSample) -> dict[str, Any]:
    if "messages" in sample:
        return dict(sample)
    for key in ("json", "data", "metadata"):
        if key in sample:
            return _decode_json_payload(sample[key])
    raise ValueError("Energon SFT samples must contain 'messages' or a JSON payload.")


def _infer_modality(member: str) -> str:
    suffix = PurePosixPath(member).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return "image"
    if suffix in {".mp4", ".webm", ".mov", ".mkv"}:
        return "video"
    if suffix in {".wav", ".flac", ".mp3", ".ogg"}:
        return "audio"
    raise ValueError(f"Cannot infer media type from member {member!r}.")


def _get_media_value(sample: CrudeSample, entry: dict[str, Any]) -> Any:
    modality = entry.get("type")
    for key in ("value", modality, "image", "video", "audio"):
        if key and key in entry:
            return entry[key]

    member = entry.get("member")
    if not isinstance(member, str) or not member:
        raise ValueError("Each media entry needs a value or shard member name.")

    sample_key = str(sample.get("__key__", ""))
    member_path = PurePosixPath(member)
    candidates = [
        member,
        member_path.name,
        member_path.suffix.removeprefix("."),
    ]
    if sample_key and member.startswith(f"{sample_key}."):
        candidates.append(member.removeprefix(f"{sample_key}."))
    for candidate in candidates:
        if candidate and candidate in sample:
            return sample[candidate]
    raise ValueError(
        f"Media member {member!r} is absent from Energon sample {sample_key!r}."
    )


@stateless
def _cook_sft_sample(sample: CrudeSample) -> CanonicalSFTSample:
    payload = _get_crude_payload(sample)
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Energon SFT samples require a non-empty messages list.")

    media: list[MediaRef] = []
    raw_media = payload.get("media", [])
    if not isinstance(raw_media, list):
        raise ValueError("The media manifest must be a list.")
    for raw_entry in raw_media:
        entry = {"member": raw_entry} if isinstance(raw_entry, str) else raw_entry
        if not isinstance(entry, dict):
            raise ValueError("Each media manifest entry must be a string or object.")
        member = entry.get("member")
        modality = entry.get("type")
        if modality is None and isinstance(member, str):
            modality = _infer_modality(member)
        if modality not in _MEDIA_TYPES:
            raise ValueError(f"Unsupported media type {modality!r}.")
        media.append(MediaRef(modality=modality, value=_get_media_value(sample, entry)))

    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError("The tools field must be a list when present.")
    return CanonicalSFTSample(
        **basic_sample_keys(sample),
        messages=deepcopy(messages),
        media=media,
        tools=deepcopy(tools),
    )


def _normalize_messages(sample: CanonicalSFTSample) -> list[dict[str, Any]]:
    messages = deepcopy(sample.messages)
    used_media: list[int] = []
    tool_call_ids: set[str] = set()

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Sample {sample.__key__!r} contains a non-object message.")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Sample {sample.__key__!r} has invalid role {role!r}.")

        content = message.get("content")
        if content is None:
            content = []
        elif isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            raise ValueError(
                f"Sample {sample.__key__!r} has unsupported content type "
                f"{type(content).__name__}."
            )

        normalized_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError(
                    f"Sample {sample.__key__!r} contains a non-object content part."
                )
            part = dict(part)
            media_index = part.pop("media_index", None)
            if media_index is not None:
                if not isinstance(media_index, int) or not 0 <= media_index < len(
                    sample.media
                ):
                    raise ValueError(
                        f"Sample {sample.__key__!r} has invalid media index "
                        f"{media_index!r}."
                    )
                media_ref = sample.media[media_index]
                declared_type = part.get("type")
                if declared_type not in (None, media_ref.modality):
                    raise ValueError(
                        f"Sample {sample.__key__!r} maps {declared_type!r} content "
                        f"to {media_ref.modality!r} media."
                    )
                part["type"] = media_ref.modality
                part[media_ref.modality] = media_ref.value
                used_media.append(media_index)
            normalized_content.append(part)
        message["content"] = normalized_content

        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict) or not isinstance(
                tool_call.get("id"), str
            ):
                raise ValueError(
                    f"Sample {sample.__key__!r} has a tool call without a string id."
                )
            tool_call_ids.add(tool_call["id"])
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or tool_call_id not in tool_call_ids:
                raise ValueError(
                    f"Sample {sample.__key__!r} has a dangling tool result id "
                    f"{tool_call_id!r}."
                )

    if used_media != list(range(len(sample.media))):
        raise ValueError(
            f"Sample {sample.__key__!r} media occurrences must be referenced once "
            f"in order; got {used_media}."
        )
    return messages


class HFMultimodalSFTProcessorAdapter:
    """Temporary Hugging Face implementation of the v1 processor boundary."""

    def __init__(
        self,
        *,
        processor: Any,
        max_sequence_length: int,
        add_bos: bool,
        add_eos: bool,
        add_generation_prompt: bool,
    ) -> None:
        if not hasattr(processor, "apply_chat_template") or not hasattr(
            processor, "tokenizer"
        ):
            raise TypeError("Energon multimodal SFT requires a Hugging Face processor.")
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_generation_prompt = add_generation_prompt
        tokenizer = processor.tokenizer
        fingerprint_data = {
            "processor_class": type(processor).__name__,
            "processor_name": getattr(processor, "name_or_path", None),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name": getattr(tokenizer, "name_or_path", None),
            "chat_template": getattr(processor, "chat_template", None)
            or getattr(tokenizer, "chat_template", None),
            "max_sequence_length": max_sequence_length,
            "add_bos": add_bos,
            "add_eos": add_eos,
            "add_generation_prompt": add_generation_prompt,
        }
        encoded = json.dumps(
            fingerprint_data, sort_keys=True, default=str
        ).encode("utf-8")
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        messages = _normalize_messages(sample)
        message_log = get_formatted_message_log(
            messages,
            self.processor,
            TaskDataSpec(),
            add_bos_token=self.add_bos,
            add_eos_token=self.add_eos,
            add_generation_prompt=self.add_generation_prompt,
            tools=sample.tools,
        )
        length = sum(len(message["token_ids"]) for message in message_log)
        loss_multiplier = 1.0
        if length >= self.max_sequence_length:
            for message in message_log:
                message["token_ids"] = message["token_ids"][
                    : min(4, self.max_sequence_length // len(message_log))
                ]
            loss_multiplier = 0.0

        model_input_keys = tuple(
            sorted(
                {
                    key
                    for message in message_log
                    for key, value in message.items()
                    if key != "token_ids"
                    and isinstance(value, (PackedTensor, torch.Tensor))
                }
            )
        )
        media_cost = sum(
            tensor.numel()
            for message in message_log
            for value in message.values()
            if isinstance(value, PackedTensor)
            for tensor in value.tensors
            if tensor is not None
        )
        media_cost_bucket = (
            0 if media_cost <= 8_000_000 else 1 if media_cost <= 64_000_000 else 2
        )
        return EncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=length,
            loss_multiplier=loss_multiplier,
            group_key=(self.fingerprint, model_input_keys, media_cost_bucket),
            sample_key=sample.__key__,
        )


class EnergonSFTTaskEncoder(
    DefaultTaskEncoder[
        CanonicalSFTSample,
        EncodedSFTSample,
        BatchedDataDict[Any],
        BatchedDataDict[Any],
    ]
):
    """Encode, group, and batch complete multimodal SFT conversations."""

    __default_failure_tolerance__ = 0
    cookers = (Cooker(_cook_sft_sample),)
    # Match the existing HF VLM path, which gives PIL RGB images to the
    # processor. The default torchrgb decoder produces already-rescaled
    # tensors that an HF image processor may rescale a second time.
    decoder = SampleDecoder(image_decode="pilrgb")

    def __init__(
        self,
        *,
        adapter: SFTProcessorAdapter,
    ) -> None:
        super().__init__()
        self.adapter = adapter

    @stateless
    def encode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        return self.adapter.encode(sample)

    def batch_group_criterion(
        self, sample: EncodedSFTSample
    ) -> tuple[tuple[Any, ...], None]:
        return sample.group_key, None

    @stateless
    def batch(self, samples: list[EncodedSFTSample]) -> BatchedDataDict[Any]:
        return BatchedDataDict(
            {
                "message_log": [sample.message_log for sample in samples],
                "loss_multiplier": torch.tensor(
                    [sample.loss_multiplier for sample in samples],
                    dtype=torch.float32,
                ),
            }
        )


def build_processor_adapter(
    *,
    processor_adapter: str,
    processor: Any,
    max_sequence_length: int,
    add_bos: bool,
    add_eos: bool,
    add_generation_prompt: bool,
) -> SFTProcessorAdapter:
    """Build the temporary v1 processor adapter."""
    if processor_adapter != "hf_multimodal":
        raise ValueError(f"Unsupported SFT processor adapter {processor_adapter!r}.")
    return HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=max_sequence_length,
        add_bos=add_bos,
        add_eos=add_eos,
        add_generation_prompt=add_generation_prompt,
    )


__all__ = [
    "CanonicalSFTSample",
    "EncodedSFTSample",
    "EnergonSFTTaskEncoder",
    "HFMultimodalSFTProcessorAdapter",
    "MediaRef",
    "SFTProcessorAdapter",
    "build_processor_adapter",
]
