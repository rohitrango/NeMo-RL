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
import logging
import os
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Protocol, cast

import torch
from megatron.energon import SampleDecoder, stateless
from PIL import Image

from nemo_rl.data.energon.multimodal.model_families import (
    ALL_MODEL_FAMILIES,
    supports_model_families,
)
from nemo_rl.data.energon.multimodal.packing import (
    pack_selected_samples,
    prepare_packed_sft_batch,
    select_samples_to_pack,
)
from nemo_rl.data.energon.multimodal.task_encoders.base import (
    BaseSFTTaskEncoder,
    SFTCooker,
)
from nemo_rl.data.energon.multimodal.task_encoders.media import (
    materialize_media_value,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
    PackedSFTSample,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.data.multimodal_utils import PackedTensor, image_patch_dim
from nemo_rl.data.packing import SequencePacker
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


logger = logging.getLogger(__name__)


def _assistant_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(content)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def add_answer_diagnostics(
    *,
    message_log: list[dict[str, Any]],
    source_messages: list[dict[str, Any]],
    tokenizer: Any,
    sample_key: str,
) -> None:
    """Mark raw answer tokens and optionally dump one decoded example."""
    if os.environ.get("NRL_SFT_ANSWER_DIAGNOSTICS") != "1":
        return
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    for message in message_log:
        tokens = message.get("token_ids")
        if isinstance(tokens, torch.Tensor):
            message["answer_token_mask"] = torch.zeros_like(tokens)
            message["answer_start_mask"] = torch.zeros_like(tokens)

    flat_tokens = torch.cat([message["token_ids"] for message in message_log])
    flat_answer_mask = torch.zeros_like(flat_tokens)
    flat_answer_start_mask = torch.zeros_like(flat_tokens)
    flat_loss_mask = torch.cat(
        [
            message.get(
                "token_loss_mask",
                torch.ones_like(message["token_ids"])
                if message.get("role") == "assistant"
                else torch.zeros_like(message["token_ids"]),
            )
            for message in message_log
        ]
    ).bool()
    cursor = 0
    answer_texts = [
        _assistant_text(message)
        for message in source_messages
        if message.get("role") == "assistant"
    ]
    for answer_text in answer_texts:
        if not answer_text:
            continue
        answer_ids = tokenizer(
            answer_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(flat_tokens.device)
        candidates = [
            start
            for start in range(cursor, len(flat_tokens) - len(answer_ids) + 1)
            if torch.equal(flat_tokens[start : start + len(answer_ids)], answer_ids)
            and bool(flat_loss_mask[start : start + len(answer_ids)].all())
        ]
        if not candidates:
            raise ValueError(
                f"Could not locate answer {answer_text!r} in supervised tokens "
                f"for sample {sample_key!r}."
            )
        start = candidates[0]
        flat_answer_mask[start : start + len(answer_ids)] = 1
        flat_answer_start_mask[start] = 1
        cursor = start + len(answer_ids)

    offset = 0
    for message in message_log:
        length = len(message["token_ids"])
        message["answer_token_mask"] = flat_answer_mask[offset : offset + length]
        message["answer_start_mask"] = flat_answer_start_mask[
            offset : offset + length
        ]
        offset += length

    dump_path = os.environ.get("NRL_SFT_ASSISTANT_TOKEN_DUMP")
    if not dump_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
    try:
        fd = os.open(dump_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    assistant_ids = flat_tokens[flat_loss_mask].cpu().tolist()
    assistant_answer_mask = flat_answer_mask[flat_loss_mask].cpu().tolist()
    payload = {
        "sample_key": sample_key,
        "answers": answer_texts,
        "decoded_assistant_response": tokenizer.decode(
            assistant_ids, clean_up_tokenization_spaces=False
        ),
        "tokens": [
            {
                "assistant_index": index,
                "token_id": token_id,
                "token": tokenizer.convert_ids_to_tokens(token_id),
                "decoded": tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                ),
                "class": "answer" if is_answer else "template",
            }
            for index, (token_id, is_answer) in enumerate(
                zip(assistant_ids, assistant_answer_mask, strict=True)
            )
        ],
    }
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def log_multimodal_diagnostic(
    *,
    processor: Any,
    sample_key: str,
    message_log: list[dict[str, Any]],
    source_image_placeholders: int,
    planned_image_positions: int | None = None,
) -> None:
    """Log token expansion and image geometry for one encoded sample."""
    tokenizer = processor.tokenizer
    token_ids = torch.cat(
        [message["token_ids"].reshape(-1).cpu() for message in message_log]
    )
    special_token_counts = {}
    for name in ("image_token", "image_break_token", "image_end_token"):
        token = getattr(processor, name, None)
        token_id = getattr(processor, f"{name}_id", None)
        if token is not None and isinstance(token_id, int):
            special_token_counts[token] = {
                "id": token_id,
                "count": int((token_ids == token_id).sum().item()),
            }

    image_sizes: list[list[int]] = []
    multimodal_shapes: dict[str, list[list[int]]] = {}
    for message in message_log:
        for key, value in message.items():
            if not isinstance(value, PackedTensor):
                continue
            tensors = [tensor for tensor in value.tensors if tensor is not None]
            multimodal_shapes.setdefault(key, []).extend(
                [list(tensor.shape) for tensor in tensors]
            )
            if key in ("imgs_sizes", "image_sizes") and not image_sizes:
                for tensor in tensors:
                    image_sizes.extend(
                        tensor.reshape(-1, 2).to(dtype=torch.int64).tolist()
                    )

    patch_size = image_patch_dim(processor)
    spatial_merge_size = int(getattr(processor, "spatial_merge_size", 1))
    merged_patch_size = patch_size * spatial_merge_size
    raw_patches = sum(
        (height // patch_size) * (width // patch_size) for height, width in image_sizes
    )
    merged_image_positions = sum(
        (height // merged_patch_size) * (width // merged_patch_size)
        for height, width in image_sizes
    )
    shown_ids = token_ids[:512].tolist()
    image_token_id = getattr(processor, "image_token_id", None)
    if not isinstance(image_token_id, int):
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    expanded_image_tokens = int((token_ids == image_token_id).sum().item())
    expected_image_positions = (
        planned_image_positions
        if planned_image_positions is not None
        else merged_image_positions
    )
    logger.info(
        "SFT multimodal diagnostic: %s",
        json.dumps(
            {
                "sample": sample_key,
                "processor": type(processor).__name__,
                "processor_name": getattr(processor, "name_or_path", None),
                "tokenizer_name": getattr(tokenizer, "name_or_path", None),
                "source_image_placeholders": source_image_placeholders,
                "processed_images": len(image_sizes),
                "placeholder_image_match": source_image_placeholders
                == len(image_sizes),
                "image_sizes": image_sizes,
                "patch_size": patch_size,
                "spatial_merge_size": spatial_merge_size,
                "raw_vision_patches": raw_patches,
                "merged_image_positions": merged_image_positions,
                "planned_image_positions": planned_image_positions,
                "expanded_image_tokens": expanded_image_tokens,
                "image_token_patch_match": expanded_image_tokens
                == expected_image_positions,
                "special_token_counts": special_token_counts,
                "multimodal_shapes": multimodal_shapes,
                "turns": [
                    {
                        "role": message["role"],
                        "token_count": len(message["token_ids"]),
                        "decoded": tokenizer.decode(
                            message["token_ids"].tolist(),
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                    }
                    for message in message_log
                ],
                "token_count": len(token_ids),
                "shown_token_count": len(shown_ids),
                "token_ids": shown_ids,
                "tokens": tokenizer.convert_ids_to_tokens(shown_ids),
                "decoded": tokenizer.decode(
                    shown_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            },
            default=str,
        ),
    )


class SFTProcessorAdapter(Protocol):
    """Boundary between canonical and model-specific SFT data."""

    @property
    def fingerprint(self) -> str: ...

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample: ...


def _normalize_messages(
    sample: CanonicalSFTSample, *, materialize: bool = True
) -> list[dict[str, Any]]:
    """Validate the message structure and attach each part's media.

    Args:
        sample: The cooked conversation.
        materialize: Decode each media value and attach the payload. Set False
            to attach the ``MediaRef`` instead.

    The Nemotron renderers replace every media part with text built from
    metadata and then overwrite ``message["content"]`` wholesale, so decoding
    for them is pure waste. It is also waste paid at the wrong time: this runs
    in pre-encode, before ``select_samples_to_pack``, so rows that selection
    discards are decoded too. Measured on video rows at 2771 ms against the
    Megatron reference's 4.7 ms, which defers all frame work to post-encode.

    Only ``GenericSFTTaskEncoder.encode`` consumes the payload, via
    ``get_formatted_message_log``, so it keeps the default.
    """
    messages = deepcopy(sample.messages)
    used_media: list[int] = []
    tool_call_ids: set[str] = set()

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(
                f"Sample {sample.__key__!r} contains a non-object message."
            )
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
                part[media_ref.modality] = (
                    materialize_media_value(
                        media_ref.value,
                        modality=media_ref.modality,
                        sample=sample,
                    )
                    if materialize
                    else media_ref
                )
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


def _zero_image_content(messages: list[dict[str, Any]]) -> None:
    """Replace image payloads with same-size black images for diagnostics."""
    for message in messages:
        for part in message["content"]:
            if part.get("type") != "image":
                continue
            image = part.get("image")
            if not isinstance(image, Image.Image):
                raise TypeError(
                    "NRL_SFT_ZERO_IMAGES requires decoded PIL image payloads."
                )
            part["image"] = Image.new(image.mode, image.size, 0)


class HFMultimodalSFTProcessorAdapter:
    """Hugging Face implementation of the generic processor boundary."""

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
        encoded = json.dumps(fingerprint_data, sort_keys=True, default=str).encode(
            "utf-8"
        )
        self._fingerprint = hashlib.sha256(encoded).hexdigest()
        self._logged_multimodal_diagnostic = False

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        messages = _normalize_messages(sample)
        if os.environ.get("NRL_SFT_ZERO_IMAGES") == "1":
            _zero_image_content(messages)
        message_log = get_formatted_message_log(
            messages,
            self.processor,
            TaskDataSpec(),
            add_bos_token=self.add_bos,
            add_eos_token=self.add_eos,
            add_generation_prompt=self.add_generation_prompt,
            tools=sample.tools,
        )
        add_answer_diagnostics(
            message_log=message_log,
            source_messages=messages,
            tokenizer=self.processor,
            sample_key=sample.__key__,
        )
        image_parts = sum(
            part.get("type") == "image"
            for message in messages
            if isinstance(message.get("content"), list)
            for part in message["content"]
        )
        if (
            os.environ.get("NRL_SFT_MULTIMODAL_DIAGNOSTICS") == "1"
            and not self._logged_multimodal_diagnostic
            and image_parts
        ):
            self._logged_multimodal_diagnostic = True
            log_multimodal_diagnostic(
                processor=self.processor,
                sample_key=sample.__key__,
                message_log=message_log,
                source_image_placeholders=image_parts,
            )
        length = sum(len(message["token_ids"]) for message in message_log)
        loss_multiplier = 1.0
        if length >= self.max_sequence_length:
            # Treat truncated messages as text only. Dropping the placeholder
            # tokens without dropping the media would leave the vision merge
            # with N features and no placeholders to scatter them into, which
            # raises rather than training a zero-weighted row.
            for message in message_log:
                message["token_ids"] = message["token_ids"][
                    : min(4, self.max_sequence_length // len(message_log))
                ]
                if "answer_token_mask" in message:
                    message["answer_token_mask"] = message["answer_token_mask"][
                        : len(message["token_ids"])
                    ]
                    message["answer_start_mask"] = message["answer_start_mask"][
                        : len(message["token_ids"])
                    ]
                for key, value in list(message.items()):
                    if isinstance(value, PackedTensor):
                        message[key] = PackedTensor.empty_like(value)
            length = sum(len(message["token_ids"]) for message in message_log)
            loss_multiplier = 0.0

        # group_key is the adapter fingerprint alone. Keying on the tensor names
        # present in message_log would split groups on any model-input difference,
        # but batch() keeps those tensors in the per-message dicts rather than in
        # a stacked batch tensor, so the finer split buys nothing.
        return EncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=length,
            packing_cost=length,
            loss_multiplier=loss_multiplier,
            group_key=(self.fingerprint,),
            sample_key=sample.__key__,
        )


@supports_model_families(ALL_MODEL_FAMILIES)
class GenericSFTTaskEncoder(BaseSFTTaskEncoder):
    """Encode, group, and batch complete multimodal SFT conversations."""

    __default_failure_tolerance__ = 0
    # Match the existing HF VLM path. Its processor expects PIL RGB images.
    decoder = SampleDecoder(image_decode="pilrgb")

    def __init__(
        self,
        *,
        adapter: SFTProcessorAdapter,
        cooker_functions: Sequence[SFTCooker],
        include_source_ids: bool,
        packer: SequencePacker | None = None,
        tokenizer: Any | None = None,
        sequence_length_pad_multiple: int = 1,
        only_unmask_final: bool = False,
        loss_mask_mode: str | None = None,
    ) -> None:
        super().__init__(cooker_functions=cooker_functions)
        self.adapter = adapter
        self.include_source_ids = include_source_ids
        self.packer = packer
        self.tokenizer = tokenizer
        self.sequence_length_pad_multiple = sequence_length_pad_multiple
        self.only_unmask_final = only_unmask_final
        self.loss_mask_mode = loss_mask_mode

    @stateless
    def preencode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        return self.adapter.encode(sample)

    @stateless
    def postencode_sample(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        return sample

    def batch_group_criterion(
        self, sample: EncodedSFTSample | PackedSFTSample
    ) -> tuple[tuple[Any, ...], None]:
        return sample.group_key, None

    @stateless(restore_seeds=True)
    def select_samples_to_pack(
        self, samples: list[EncodedSFTSample]
    ) -> list[list[EncodedSFTSample]]:
        if self.packer is None:
            raise RuntimeError("Energon packing is not configured.")
        packs = select_samples_to_pack(
            samples,
            packer=self.packer,
            sequence_length_pad_multiple=self.sequence_length_pad_multiple,
        )
        random.shuffle(packs)
        return packs

    @stateless
    def pack_selected_samples(self, samples: list[EncodedSFTSample]) -> PackedSFTSample:
        if self.packer is None:
            raise RuntimeError("Energon packing is not configured.")
        return pack_selected_samples(
            samples,
            pack_capacity=self.packer.bin_capacity,
            sequence_length_pad_multiple=self.sequence_length_pad_multiple,
        )

    @stateless
    def batch(
        self, samples: list[EncodedSFTSample | PackedSFTSample]
    ) -> BatchedDataDict[Any]:
        if samples and isinstance(samples[0], PackedSFTSample):
            if not all(isinstance(sample, PackedSFTSample) for sample in samples):
                raise TypeError("Energon batches cannot mix packed and unpacked rows.")
            if self.tokenizer is None:
                raise RuntimeError("Packed SFT requires a tokenizer.")
            return prepare_packed_sft_batch(
                cast(list[PackedSFTSample], samples),
                tokenizer=self.tokenizer,
                only_unmask_final=self.only_unmask_final,
                loss_mask_mode=self.loss_mask_mode,
            )
        if not all(isinstance(sample, EncodedSFTSample) for sample in samples):
            raise TypeError("Energon SFT batches accept only encoded samples.")
        encoded_samples = cast(list[EncodedSFTSample], samples)
        values: dict[str, Any] = {
            "message_log": [sample.message_log for sample in encoded_samples],
            "loss_multiplier": torch.tensor(
                [sample.loss_multiplier for sample in encoded_samples],
                dtype=torch.float32,
            ),
        }
        if self.include_source_ids:
            values["source_ids"] = [sample.sample_key for sample in encoded_samples]
        if self.loss_mask_mode is not None:
            values["loss_mask_mode"] = self.loss_mask_mode
        return BatchedDataDict(values)

    @stateless
    def encode_batch(self, batch: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        return batch


def build_processor_adapter(
    *,
    processor_adapter: str,
    processor: Any,
    max_sequence_length: int,
    add_bos: bool,
    add_eos: bool,
    add_generation_prompt: bool,
) -> SFTProcessorAdapter:
    """Build the configured model processor adapter."""
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
    "GenericSFTTaskEncoder",
    "HFMultimodalSFTProcessorAdapter",
    "SFTProcessorAdapter",
    "build_processor_adapter",
]
