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
from difflib import SequenceMatcher
from pathlib import PurePosixPath
from typing import Any, Protocol

import torch
import torch.nn.functional as F
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

from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    extract_multimodal_model_inputs,
    uses_image_placeholder,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

_MEDIA_TYPES = frozenset({"image", "video", "audio"})
_ASSISTANT_MASK_KEYS = ("assistant_masks", "assistant_mask")


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
    """One processor-encoded conversation before batching."""

    input_ids: torch.Tensor
    token_mask: torch.Tensor
    model_inputs: dict[str, PackedTensor | torch.Tensor]
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


def _as_single_row(value: Any, *, key: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"Processor output is missing {key!r}.")
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(f"Processor field {key!r} must contain one conversation.")
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(f"Processor field {key!r} must be one-dimensional.")
    return tensor


def _assistant_mask(processed: dict[str, Any]) -> torch.Tensor:
    for key in _ASSISTANT_MASK_KEYS:
        if key in processed:
            return _as_single_row(processed[key], key=key).to(dtype=torch.long)
    raise ValueError(
        "The processor did not return an assistant mask. Its chat template must "
        "mark assistant generations and use a fast tokenizer."
    )


def _last_assistant_only(mask: torch.Tensor) -> torch.Tensor:
    trainable = torch.nonzero(mask, as_tuple=False).flatten()
    if len(trainable) == 0:
        return mask
    gaps = torch.nonzero(trainable[1:] > trainable[:-1] + 1, as_tuple=False).flatten()
    start = trainable[gaps[-1] + 1] if len(gaps) else trainable[0]
    result = torch.zeros_like(mask)
    result[start : trainable[-1] + 1] = mask[start : trainable[-1] + 1]
    return result


def _align_expanded_mask(
    raw_ids: torch.Tensor,
    raw_mask: torch.Tensor,
    expanded_ids: torch.Tensor,
) -> torch.Tensor:
    if torch.equal(raw_ids, expanded_ids):
        return raw_mask
    aligned = torch.zeros_like(expanded_ids, dtype=torch.long)
    matcher = SequenceMatcher(
        a=raw_ids.tolist(), b=expanded_ids.tolist(), autojunk=False
    )
    for tag, raw_start, raw_end, out_start, out_end in matcher.get_opcodes():
        if tag == "equal":
            aligned[out_start:out_end] = raw_mask[raw_start:raw_end]
        elif raw_end > raw_start:
            replacement_mask = raw_mask[raw_start:raw_end]
            if bool(torch.all(replacement_mask == replacement_mask[0])):
                aligned[out_start:out_end] = replacement_mask[0]
    return aligned


def _placeholder_message(message: dict[str, Any], processor: Any) -> dict[str, Any]:
    if hasattr(processor, "conversation_preprocessor"):
        return processor.conversation_preprocessor(message)

    tokens = {
        "image": getattr(processor, "image_token", "<image>"),
        "video": getattr(processor, "video_token", "<video>"),
        "audio": getattr(processor, "audio_token", "<audio>"),
    }
    text_parts: list[str] = []
    for part in message["content"]:
        part_type = part.get("type")
        if part_type == "text":
            text_parts.append(str(part.get("text", "")))
        elif part_type in tokens:
            text_parts.append(tokens[part_type])
    result = dict(message)
    result["content"] = "\n".join(text_parts)
    return result


def _collect_media(messages: list[dict[str, Any]]) -> dict[str, list[Any]]:
    media = {"images": [], "videos": [], "audio": []}
    output_key = {"image": "images", "video": "videos", "audio": "audio"}
    for message in messages:
        for part in message["content"]:
            part_type = part.get("type")
            if part_type in output_key:
                value = part.get(part_type)
                if value is None:
                    value = part.get("url")
                if value is None:
                    value = part.get("path")
                if value is None:
                    raise ValueError(f"{part_type!r} content is missing its value.")
                media[output_key[part_type]].append(value)
    return media


class HFMultimodalSFTProcessorAdapter:
    """Temporary Hugging Face implementation of the v1 processor boundary."""

    def __init__(
        self,
        *,
        processor: Any,
        max_sequence_length: int,
        only_unmask_final: bool,
    ) -> None:
        if not hasattr(processor, "apply_chat_template") or not hasattr(
            processor, "tokenizer"
        ):
            raise TypeError("Energon multimodal SFT requires a Hugging Face processor.")
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.only_unmask_final = only_unmask_final
        tokenizer = processor.tokenizer
        fingerprint_data = {
            "processor_class": type(processor).__name__,
            "processor_name": getattr(processor, "name_or_path", None),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name": getattr(tokenizer, "name_or_path", None),
            "chat_template": getattr(processor, "chat_template", None)
            or getattr(tokenizer, "chat_template", None),
            "max_sequence_length": max_sequence_length,
            "only_unmask_final": only_unmask_final,
        }
        encoded = json.dumps(
            fingerprint_data, sort_keys=True, default=str
        ).encode("utf-8")
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": False,
            "return_tensors": "pt",
            "return_dict": True,
            "return_assistant_tokens_mask": True,
        }
        if tools is not None:
            template_kwargs["tools"] = tools
        return dict(self.processor.apply_chat_template(messages, **template_kwargs))

    def _apply_placeholder_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        template_messages = [
            _placeholder_message(message, self.processor) for message in messages
        ]
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": False,
            "return_tensors": "pt",
            "return_dict": True,
            "return_assistant_tokens_mask": True,
        }
        render_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        if tools is not None:
            template_kwargs["tools"] = tools
            render_kwargs["tools"] = tools
        raw = dict(
            self.processor.tokenizer.apply_chat_template(
                template_messages, **template_kwargs
            )
        )
        rendered = self.processor.apply_chat_template(
            template_messages, **render_kwargs
        )
        media = _collect_media(messages)
        processor_kwargs = {
            key: value for key, value in media.items() if value
        }
        processed = dict(
            self.processor(text=rendered, return_tensors="pt", **processor_kwargs)
        )
        raw_ids = _as_single_row(raw["input_ids"], key="input_ids")
        raw_mask = _assistant_mask(raw)
        expanded_ids = _as_single_row(processed["input_ids"], key="input_ids")
        processed["assistant_masks"] = _align_expanded_mask(
            raw_ids, raw_mask, expanded_ids
        ).unsqueeze(0)
        return processed

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        messages = _normalize_messages(sample)
        if uses_image_placeholder(self.processor):
            processed = self._apply_placeholder_template(messages, sample.tools)
        else:
            processed = self._apply_chat_template(messages, sample.tools)

        input_ids = _as_single_row(processed.get("input_ids"), key="input_ids")
        token_mask = _assistant_mask(processed)
        if len(input_ids) != len(token_mask):
            raise ValueError(
                f"Sample {sample.__key__!r} has {len(input_ids)} tokens but "
                f"{len(token_mask)} assistant-mask entries."
            )
        if len(input_ids) > self.max_sequence_length:
            raise ValueError(
                f"Sample {sample.__key__!r} has {len(input_ids)} tokens, exceeding "
                f"the configured maximum {self.max_sequence_length}."
            )
        if self.only_unmask_final:
            token_mask = _last_assistant_only(token_mask)
        if not bool(torch.any(token_mask)):
            raise ValueError(
                f"Sample {sample.__key__!r} has no trainable assistant tokens."
            )

        model_inputs = extract_multimodal_model_inputs(self.processor, processed)
        media_cost = sum(
            tensor.numel()
            for value in model_inputs.values()
            if isinstance(value, PackedTensor)
            for tensor in value.tensors
            if tensor is not None
        )
        media_cost_bucket = (
            0 if media_cost <= 8_000_000 else 1 if media_cost <= 64_000_000 else 2
        )
        token_sidecars = tuple(
            sorted(
                key
                for key, value in model_inputs.items()
                if isinstance(value, torch.Tensor)
            )
        )
        return EncodedSFTSample.derive_from(
            sample,
            input_ids=input_ids.to(dtype=torch.long),
            token_mask=token_mask.to(dtype=torch.long),
            model_inputs=model_inputs,
            group_key=(self.fingerprint, token_sidecars, media_cost_bucket),
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
        pad_token_id: int,
        sequence_length_pad_multiple: int,
    ) -> None:
        super().__init__()
        if sequence_length_pad_multiple < 1:
            raise ValueError("sequence_length_pad_multiple must be positive.")
        self.adapter = adapter
        self.pad_token_id = pad_token_id
        self.sequence_length_pad_multiple = sequence_length_pad_multiple

    @stateless
    def encode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        return self.adapter.encode(sample)

    def batch_group_criterion(
        self, sample: EncodedSFTSample
    ) -> tuple[tuple[Any, ...], None]:
        return sample.group_key, None

    def _sample_batch(self, sample: EncodedSFTSample) -> dict[str, Any]:
        sequence_length = len(sample.input_ids)
        padded_length = (
            (sequence_length + self.sequence_length_pad_multiple - 1)
            // self.sequence_length_pad_multiple
            * self.sequence_length_pad_multiple
        )
        pad_length = padded_length - sequence_length
        batch: dict[str, Any] = {
            "input_ids": F.pad(
                sample.input_ids,
                (0, pad_length),
                value=self.pad_token_id,
            ).unsqueeze(0),
            "input_lengths": torch.tensor([sequence_length], dtype=torch.long),
            "token_mask": F.pad(sample.token_mask, (0, pad_length)).unsqueeze(0),
            "sample_mask": torch.ones(1, dtype=torch.float32),
        }
        for key, value in sample.model_inputs.items():
            if isinstance(value, PackedTensor):
                batch[key] = value
            else:
                batch[key] = F.pad(value, (0, pad_length)).unsqueeze(0)
        return batch

    @stateless
    def batch(self, samples: list[EncodedSFTSample]) -> BatchedDataDict[Any]:
        return BatchedDataDict.from_batches(
            [self._sample_batch(sample) for sample in samples],
            pad_value_dict={"input_ids": self.pad_token_id},
            allow_missing_packed_tensors=True,
        )


def build_processor_adapter(
    *,
    processor_adapter: str,
    processor: Any,
    max_sequence_length: int,
    only_unmask_final: bool,
) -> SFTProcessorAdapter:
    """Build the temporary v1 processor adapter."""
    if processor_adapter != "hf_multimodal":
        raise ValueError(f"Unsupported SFT processor adapter {processor_adapter!r}.")
    return HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=max_sequence_length,
        only_unmask_final=only_unmask_final,
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
