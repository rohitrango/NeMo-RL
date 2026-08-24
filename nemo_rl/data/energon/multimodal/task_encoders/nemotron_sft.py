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
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from typing import Any

import torch
from megatron.energon import stateless
from PIL import Image

from nemo_rl.data.energon.multimodal.model_families import supports_model_families
from nemo_rl.data.energon.multimodal.packing import EnergonPackingHooks
from nemo_rl.data.energon.multimodal.task_encoders.base import SFTCooker
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    GenericSFTTaskEncoder,
    HFMultimodalSFTProcessorAdapter,
    SFTProcessorAdapter,
    _normalize_messages,
)
from nemo_rl.data.energon.multimodal.task_encoders.media import (
    decode_selected_av_bytes,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
    MediaRef,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    extract_multimodal_model_inputs,
    resolve_to_image,
)

COMPACT_IMAGE_PLACEHOLDER = "<img><image></img>"
_VISUAL_MODEL_INPUT_KEYS = ("imgs_sizes", "num_frames", "pixel_values")

# Thinking-tag spacing follows the Nemotron rules in Megatron-LM revision
# 6822175d92a40e0528be905aee50f5930cfa0c98, multimodal/data_loading/task_encoder.py.


def _metadata(ref: MediaRef) -> dict[str, str | int | float | bool | None]:
    return dict(ref.metadata)


def _required_metadata_int(ref: MediaRef, key: str) -> int:
    value = _metadata(ref).get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"Nemotron {ref.modality} media requires positive integer metadata "
            f"{key!r}; got {value!r}."
        )
    return value


def _required_metadata_float(ref: MediaRef, key: str) -> float:
    value = _metadata(ref).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            f"Nemotron {ref.modality} media requires positive numeric metadata "
            f"{key!r}; got {value!r}."
        )
    return float(value)


def _radio_token_count(*, height: int, width: int, patch_dim: int) -> int:
    """Return RADIO features after the fixed 2x2 spatial pixel shuffle.

    This matches ``_pixel_shuffled_token_count`` in the Apache-2.0
    Megatron-Bridge source at revision 8c46dc4259080c510b7455f43e836fdff222c5d3.
    """
    if patch_dim <= 0:
        raise ValueError("patch_dim must be greater than zero.")
    if height % patch_dim or width % patch_dim:
        raise ValueError(
            f"Processed image {height}x{width} is not divisible by "
            f"patch_dim={patch_dim}."
        )
    patch_rows = height // patch_dim
    patch_columns = width // patch_dim
    if patch_rows % 2 or patch_columns % 2:
        raise ValueError(
            f"Processed image {height}x{width} produces a patch grid that is "
            "not divisible by the 2x2 RADIO pixel shuffle."
        )
    return patch_rows * patch_columns // 4


def _video_prompt(
    ref: MediaRef,
    *,
    temporal_patch_size: int,
) -> tuple[str, int]:
    """Build the compact temporal prompt without reading the video payload.

    The line layout matches ``_prepare_temporal_rows`` in the pinned Apache-2.0
    Megatron-Bridge Nemotron Omni collator.
    """
    frame_count = _required_metadata_int(ref, "sampled_num_frames")
    sampled_fps = _required_metadata_float(ref, "sampled_fps")
    tubelet_count = math.ceil(frame_count / temporal_patch_size)
    lines = ["This is a video:"]
    for frame_start in range(0, frame_count, temporal_patch_size):
        frame_end = min(frame_start + temporal_patch_size, frame_count)
        timestamps = [
            f"{'Frame' if frame_index == frame_start else 'frame'} "
            f"{frame_index + 1} sampled at {frame_index / sampled_fps:.2f} seconds"
            for frame_index in range(frame_start, frame_end)
        ]
        lines.append(" and ".join(timestamps) + f": {COMPACT_IMAGE_PLACEHOLDER}")
    return "\n".join(lines), tubelet_count


def _predicted_visual_cost(
    ref: MediaRef,
    *,
    patch_dim: int,
    temporal_patch_size: int,
) -> tuple[int, int]:
    height = _required_metadata_int(ref, "processed_height")
    width = _required_metadata_int(ref, "processed_width")
    tokens_per_tile = _radio_token_count(
        height=height,
        width=width,
        patch_dim=patch_dim,
    )
    if ref.modality == "image":
        num_tiles = _required_metadata_int(ref, "num_tiles")
        return num_tiles * tokens_per_tile, 1
    if ref.modality == "video":
        frame_count = _required_metadata_int(ref, "sampled_num_frames")
        tubelet_count = math.ceil(frame_count / temporal_patch_size)
        return tubelet_count * tokens_per_tile, tubelet_count
    raise ValueError(
        f"Nemotron visual encoding supports image and video media, got "
        f"{ref.modality!r}."
    )


def _clean_thinking_trace(match: re.Match[str], *, ultra_format: bool) -> str:
    content = match.group(1).strip()
    if not content:
        return "<think></think>"
    closing_prefix = "" if ultra_format else "\n"
    return f"<think>\n{content}{closing_prefix}</think>"


def _normalize_assistant_thinking(
    message: dict[str, Any],
    *,
    prompt_format: str,
    thinking_trace_format: str,
) -> None:
    if message["role"] != "assistant":
        return
    content = message["content"]
    if any(part["type"] != "text" for part in content):
        raise ValueError("Nemotron assistant turns cannot contain visual media.")
    text = "".join(str(part.get("text", "")) for part in content)
    start_count = text.count("<think>")
    end_count = text.count("</think>")
    if start_count == 0 and end_count == 0:
        prefix = (
            "<think></think>"
            if prompt_format == "nemotron6-moe"
            else "<think></think>\n\n"
        )
        message["content"] = [{"type": "text", "text": prefix + text.strip()}]
        return
    if start_count != 1 or end_count != 1:
        raise ValueError(
            "Nemotron assistant turns require exactly one matched pair of <think> tags."
        )
    if text.find("<think>") > text.find("</think>"):
        raise ValueError("Nemotron assistant </think> appears before <think>.")

    ultra_format = thinking_trace_format == "ultra"
    normalized = re.sub(
        r"<think>(.*?)</think>",
        lambda match: _clean_thinking_trace(match, ultra_format=ultra_format),
        text,
        flags=re.DOTALL,
    )
    if ultra_format:
        separator = "</think>"
    elif prompt_format == "nemotron6-moe":
        separator = "</think>\n"
    else:
        separator = "</think>\n\n"
    normalized = re.sub(r"</think>\s*", separator, normalized)
    message["content"] = [{"type": "text", "text": normalized}]


def _render_compact_messages(
    sample: CanonicalSFTSample,
    *,
    temporal_patch_size: int,
    prompt_format: str,
    thinking_trace_format: str,
) -> tuple[list[dict[str, Any]], list[tuple[int, MediaRef]]]:
    messages = _normalize_messages(sample)
    occurrences: list[tuple[int, MediaRef]] = []
    media_index = 0
    for message_index, message in enumerate(messages):
        _normalize_assistant_thinking(
            message,
            prompt_format=prompt_format,
            thinking_trace_format=thinking_trace_format,
        )
        content = message["content"]
        rendered_content: list[dict[str, str]] = []
        for part in content:
            part_type = part["type"]
            if part_type == "text":
                rendered_content.append(
                    {"type": "text", "text": str(part.get("text", ""))}
                )
                continue
            if part_type not in {"image", "video"}:
                raise ValueError(
                    f"Nemotron visual encoding does not support {part_type!r} media."
                )
            if media_index >= len(sample.media):
                raise ValueError(
                    "Nemotron message media exceeds the canonical media list."
                )
            ref = sample.media[media_index]
            if ref.modality != part_type:
                raise ValueError(
                    f"Nemotron media order mismatch: expected {part_type!r}, got "
                    f"{ref.modality!r}."
                )
            if ref.modality == "image":
                placeholder = COMPACT_IMAGE_PLACEHOLDER
            else:
                placeholder, _ = _video_prompt(
                    ref,
                    temporal_patch_size=temporal_patch_size,
                )
            rendered_content.append({"type": "text", "text": placeholder})
            occurrences.append((message_index, ref))
            media_index += 1
        message["content"] = rendered_content
    if media_index != len(sample.media):
        raise ValueError(
            f"Nemotron messages reference {media_index}/{len(sample.media)} media items."
        )
    return messages, occurrences


def _packed_tensor_value(
    model_inputs: dict[str, PackedTensor | torch.Tensor],
    key: str,
) -> torch.Tensor:
    value = model_inputs.get(key)
    if not isinstance(value, PackedTensor):
        raise ValueError(
            f"Nemotron processor output must include packed {key!r} metadata."
        )
    tensor = value.as_tensor()
    if tensor is None:
        raise ValueError(f"Nemotron processor output {key!r} is empty.")
    return tensor


def _decode_selected_payload(ref: MediaRef) -> Any:
    """Decode only media that survived pack selection."""
    value = ref.value
    if ref.modality == "image":
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(bytes(value))) as image:
                return image.convert("RGB")
        if isinstance(value, str):
            return resolve_to_image(value)
        return value
    if ref.modality == "video" and isinstance(value, (bytes, bytearray, memoryview)):
        return decode_selected_av_bytes(value, modality="video")
    return value


def _token_id(tokenizer: Any, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise TypeError("Nemotron tokenizer must expose convert_tokens_to_ids().")
    token_id = convert(token)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if (
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id == unknown_id
    ):
        raise ValueError(f"Nemotron tokenizer does not define token {token!r}.")
    return token_id


def _expand_visual_placeholders(
    message_log: list[dict[str, Any]],
    occurrences: Sequence[tuple[int, tuple[int, ...]]],
    *,
    image_token_id: int,
) -> None:
    """Replace each compact visual token with its measured feature width."""
    widths_by_message: defaultdict[int, list[int]] = defaultdict(list)
    for message_index, widths in occurrences:
        widths_by_message[message_index].extend(widths)

    for message_index, widths in widths_by_message.items():
        token_ids = message_log[message_index].get("token_ids")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError(
                "Nemotron tokenized messages require one-dimensional token_ids."
            )
        placeholder_positions = torch.where(token_ids == image_token_id)[0].tolist()
        if len(placeholder_positions) != len(widths):
            raise ValueError(
                f"Nemotron message {message_index} has "
                f"{len(placeholder_positions)} compact visual placeholders for "
                f"{len(widths)} projected visual rows."
            )

        pieces: list[torch.Tensor] = []
        start = 0
        for position, width in zip(placeholder_positions, widths, strict=True):
            if width <= 0:
                raise ValueError("Nemotron visual expansion widths must be positive.")
            pieces.append(token_ids[start:position])
            pieces.append(
                torch.full(
                    (width,),
                    image_token_id,
                    dtype=token_ids.dtype,
                    device=token_ids.device,
                )
            )
            start = position + 1
        pieces.append(token_ids[start:])
        message_log[message_index]["token_ids"] = torch.cat(pieces)


class NemotronVisualSFTProcessorAdapter:
    """Split Nemotron visual cost prediction from media processing.

    Image rows must provide ``processed_height``, ``processed_width``, and
    ``num_tiles`` metadata. Video rows must provide ``processed_height``,
    ``processed_width``, ``sampled_num_frames``, and ``sampled_fps`` metadata.
    These fields describe selected processor output, not raw media dimensions.
    """

    def __init__(
        self,
        *,
        processor: Any,
        max_sequence_length: int,
        patch_dim: int = 16,
        temporal_patch_size: int = 2,
        prompt_format: str = "nemotron-h-5p5-reasoning",
        thinking_trace_format: str = "default",
        add_bos: bool = False,
        add_eos: bool = False,
        add_generation_prompt: bool = False,
    ) -> None:
        if patch_dim <= 0:
            raise ValueError("patch_dim must be greater than zero.")
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be greater than zero.")
        if prompt_format not in {"nemotron-h-5p5-reasoning", "nemotron6-moe"}:
            raise ValueError(f"Unsupported Nemotron prompt format {prompt_format!r}.")
        if thinking_trace_format not in {"default", "ultra"}:
            raise ValueError(
                f"Unsupported Nemotron thinking trace format {thinking_trace_format!r}."
            )
        if not hasattr(processor, "apply_chat_template") or not hasattr(
            processor, "tokenizer"
        ):
            raise TypeError("Nemotron visual SFT requires a Hugging Face processor.")
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.patch_dim = patch_dim
        self.temporal_patch_size = temporal_patch_size
        self.prompt_format = prompt_format
        self.thinking_trace_format = thinking_trace_format
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_generation_prompt = add_generation_prompt
        fingerprint_data = {
            "processor_class": type(processor).__name__,
            "processor_name": getattr(processor, "name_or_path", None),
            "patch_dim": patch_dim,
            "temporal_patch_size": temporal_patch_size,
            "prompt_format": prompt_format,
            "thinking_trace_format": thinking_trace_format,
            "max_sequence_length": max_sequence_length,
            "add_bos": add_bos,
            "add_eos": add_eos,
            "add_generation_prompt": add_generation_prompt,
        }
        encoded = json.dumps(fingerprint_data, sort_keys=True, default=str).encode()
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Provide the generic adapter method as the lazy pre-encode operation."""
        return self.preencode(sample)

    def preencode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Tokenize compact placeholders and predict expanded cost from metadata."""
        messages, occurrences = _render_compact_messages(
            sample,
            temporal_patch_size=self.temporal_patch_size,
            prompt_format=self.prompt_format,
            thinking_trace_format=self.thinking_trace_format,
        )
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
        visual_embeddings = 0
        compact_placeholders = 0
        for _, ref in occurrences:
            embeddings, placeholders = _predicted_visual_cost(
                ref,
                patch_dim=self.patch_dim,
                temporal_patch_size=self.temporal_patch_size,
            )
            visual_embeddings += embeddings
            compact_placeholders += placeholders
        packing_cost = length + visual_embeddings - compact_placeholders
        if packing_cost > self.max_sequence_length:
            raise ValueError(
                f"Nemotron sample {sample.__key__!r} has expanded length "
                f"{packing_cost}, above max_sequence_length={self.max_sequence_length}."
            )
        cost_bucket = (
            0 if visual_embeddings <= 256 else 1 if visual_embeddings <= 2_048 else 2
        )
        model_input_keys = _VISUAL_MODEL_INPUT_KEYS if occurrences else ()
        return EncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=length,
            packing_cost=packing_cost,
            loss_multiplier=1.0,
            group_key=(self.fingerprint, model_input_keys, cost_bucket),
            sample_key=sample.__key__,
            pending_sample=sample,
        )

    def _process_media(
        self,
        ref: MediaRef,
    ) -> tuple[dict[str, PackedTensor], int]:
        predicted_embeddings, _ = _predicted_visual_cost(
            ref,
            patch_dim=self.patch_dim,
            temporal_patch_size=self.temporal_patch_size,
        )
        if ref.modality == "image":
            text = COMPACT_IMAGE_PLACEHOLDER
            media_kwargs = {"images": [_decode_selected_payload(ref)]}
        elif ref.modality == "video":
            text, _ = _video_prompt(
                ref,
                temporal_patch_size=self.temporal_patch_size,
            )
            media_kwargs = {"videos": [_decode_selected_payload(ref)]}
        else:
            raise ValueError(
                f"Nemotron visual encoding does not support {ref.modality!r} media."
            )

        processed = dict(
            self.processor(
                text=[text],
                return_tensors="pt",
                add_special_tokens=False,
                **media_kwargs,
            )
        )
        if ref.modality == "video" and "num_frames" not in processed:
            raise ValueError("Nemotron video processor output must include num_frames.")
        model_inputs = extract_multimodal_model_inputs(self.processor, processed)
        unexpected_sequence_inputs = {
            key: value
            for key, value in model_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        if unexpected_sequence_inputs:
            raise ValueError(
                "Nemotron visual post-encoding does not support per-media sequence "
                f"inputs: {sorted(unexpected_sequence_inputs)!r}."
            )
        packed_inputs = {
            key: value
            for key, value in model_inputs.items()
            if isinstance(value, PackedTensor)
        }
        if "pixel_values" not in packed_inputs:
            raise ValueError("Nemotron processor output is missing pixel_values.")

        sizes = _packed_tensor_value(model_inputs, "imgs_sizes").reshape(-1, 2)
        frame_counts = _packed_tensor_value(model_inputs, "num_frames").reshape(-1)
        if ref.modality == "image":
            expected_tiles = _required_metadata_int(ref, "num_tiles")
            if (
                len(sizes) != expected_tiles
                or len(frame_counts) != expected_tiles
                or not bool((frame_counts == 1).all())
            ):
                raise ValueError(
                    "Nemotron image processor output does not match predicted tile "
                    f"metadata: got {len(sizes)} sizes and num_frames="
                    f"{frame_counts.tolist()}, expected {expected_tiles} image tiles."
                )
            actual_embeddings = sum(
                _radio_token_count(
                    height=int(height),
                    width=int(width),
                    patch_dim=self.patch_dim,
                )
                for height, width in sizes.tolist()
            )
        else:
            expected_frames = _required_metadata_int(ref, "sampled_num_frames")
            if len(frame_counts) != 1 or int(frame_counts[0]) != expected_frames:
                raise ValueError(
                    "Nemotron video processor output does not match predicted frame "
                    f"metadata: got num_frames={frame_counts.tolist()}, expected "
                    f"[{expected_frames}]."
                )
            if len(sizes) != expected_frames:
                raise ValueError(
                    f"Nemotron video processor returned {len(sizes)} frame sizes for "
                    f"{expected_frames} frames."
                )
            per_frame_counts = [
                _radio_token_count(
                    height=int(height),
                    width=int(width),
                    patch_dim=self.patch_dim,
                )
                for height, width in sizes.tolist()
            ]
            if len(set(per_frame_counts)) != 1:
                raise ValueError(
                    "Nemotron temporal tubelets require one processed frame size."
                )
            tubelet_count = math.ceil(expected_frames / self.temporal_patch_size)
            actual_embeddings = tubelet_count * per_frame_counts[0]

        if actual_embeddings != predicted_embeddings:
            raise ValueError(
                f"Nemotron {ref.modality} expansion changed after media processing: "
                f"predicted {predicted_embeddings}, got {actual_embeddings}."
            )
        return packed_inputs, actual_embeddings

    def postencode(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        """Process selected media and verify its measured expansion cost."""
        pending_sample = sample.pending_sample
        if pending_sample is None:
            raise ValueError(
                f"Nemotron sample {sample.sample_key!r} has no pending canonical media."
            )
        _, occurrences = _render_compact_messages(
            pending_sample,
            temporal_patch_size=self.temporal_patch_size,
            prompt_format=self.prompt_format,
            thinking_trace_format=self.thinking_trace_format,
        )
        message_log = deepcopy(sample.message_log)
        inputs_by_message: defaultdict[int, defaultdict[str, list[PackedTensor]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        visual_occurrences: list[tuple[int, tuple[int, ...]]] = []
        actual_embeddings = 0
        compact_placeholders = 0
        for message_index, ref in occurrences:
            media_inputs, media_embeddings = self._process_media(ref)
            for key, value in media_inputs.items():
                inputs_by_message[message_index][key].append(value)
            actual_embeddings += media_embeddings
            _, placeholders = _predicted_visual_cost(
                ref,
                patch_dim=self.patch_dim,
                temporal_patch_size=self.temporal_patch_size,
            )
            compact_placeholders += placeholders
            if media_embeddings % placeholders:
                raise ValueError(
                    f"Nemotron {ref.modality} produced {media_embeddings} visual "
                    f"features for {placeholders} placeholder rows."
                )
            visual_occurrences.append(
                (message_index, (media_embeddings // placeholders,) * placeholders)
            )

        actual_cost = sample.length + actual_embeddings - compact_placeholders
        if actual_cost != sample.packing_cost:
            raise ValueError(
                f"Nemotron sample {sample.sample_key!r} expanded length changed "
                f"after media processing: predicted {sample.packing_cost}, got "
                f"{actual_cost}."
            )
        image_token_id = _token_id(self.processor.tokenizer, "<image>")
        _expand_visual_placeholders(
            message_log,
            visual_occurrences,
            image_token_id=image_token_id,
        )
        expanded_length = sum(len(message["token_ids"]) for message in message_log)
        if expanded_length != sample.packing_cost:
            raise ValueError(
                f"Nemotron sample {sample.sample_key!r} expanded to "
                f"{expanded_length} token ids, expected {sample.packing_cost}."
            )
        for message_index, keyed_inputs in inputs_by_message.items():
            message_log[message_index].update(
                {
                    key: PackedTensor.merge_segments(values)
                    for key, values in keyed_inputs.items()
                }
            )
        return EncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=expanded_length,
            pending_sample=None,
        )


@supports_model_families("nemotron")
class NemotronSFTTaskEncoder(GenericSFTTaskEncoder):
    """Run the split Nemotron visual SFT encoding lifecycle."""

    # Keep WDS payloads lazy until postencode selects the sample.
    decoder = None

    def __init__(
        self,
        *,
        adapter: SFTProcessorAdapter,
        cooker_functions: Sequence[SFTCooker],
        packing_hooks: EnergonPackingHooks[Any, Any, Any] | None,
        include_source_ids: bool,
        patch_dim: int = 16,
        temporal_patch_size: int = 2,
        prompt_format: str = "nemotron-h-5p5-reasoning",
        thinking_trace_format: str = "default",
    ) -> None:
        if isinstance(adapter, NemotronVisualSFTProcessorAdapter):
            visual_adapter = adapter
        elif isinstance(adapter, HFMultimodalSFTProcessorAdapter):
            visual_adapter = NemotronVisualSFTProcessorAdapter(
                processor=adapter.processor,
                max_sequence_length=adapter.max_sequence_length,
                patch_dim=patch_dim,
                temporal_patch_size=temporal_patch_size,
                prompt_format=prompt_format,
                thinking_trace_format=thinking_trace_format,
                add_bos=adapter.add_bos,
                add_eos=adapter.add_eos,
                add_generation_prompt=adapter.add_generation_prompt,
            )
        else:
            raise TypeError(
                "Nemotron SFT requires the Hugging Face or Nemotron visual "
                "processor adapter."
            )
        super().__init__(
            adapter=visual_adapter,
            cooker_functions=cooker_functions,
            packing_hooks=packing_hooks,
            include_source_ids=include_source_ids,
        )
        self._visual_adapter = visual_adapter

    @stateless
    def preencode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        return self._visual_adapter.preencode(sample)

    @stateless
    def postencode_sample(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        return self._visual_adapter.postencode(sample)


__all__ = [
    "COMPACT_IMAGE_PLACEHOLDER",
    "NemotronSFTTaskEncoder",
    "NemotronVisualSFTProcessorAdapter",
]
