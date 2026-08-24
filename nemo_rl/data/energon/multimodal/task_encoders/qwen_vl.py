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

import math
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

import torch
from megatron.energon import stateless
from PIL import Image

from nemo_rl.data.energon.multimodal.model_families import supports_model_families
from nemo_rl.data.energon.multimodal.packing import EnergonPackingHooks
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    GenericSFTTaskEncoder,
    HFMultimodalSFTProcessorAdapter,
    SFTProcessorAdapter,
    _normalize_messages,
)
from nemo_rl.data.energon.multimodal.task_encoders.base import SFTCooker
from nemo_rl.data.energon.multimodal.task_encoders.media import (
    decode_selected_av_bytes,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
    MediaRef,
)
from nemo_rl.data.llm_message_utils import get_first_index_that_differs
from nemo_rl.data.multimodal_utils import PackedTensor

# Qwen resize and grid accounting follow Megatron-Bridge
# 8c46dc4259080c510b7455f43e836fdff222c5d3, models/qwen_vl/data/energon.py,
# and the Qwen processor settings selected by the Hugging Face processor.


def _positive_int(value: object, *, name: str, sample_key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Qwen sample {sample_key!r} requires positive integer media metadata "
            f"{name!r}."
        )
    return value


def _metadata_value(media: MediaRef, *names: str) -> object:
    metadata = dict(media.metadata)
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def _processor_setting(processor: Any, modality: str, *names: str) -> object:
    component = getattr(processor, f"{modality}_processor", None)
    sources = [component, processor]
    for source in sources:
        if source is None:
            continue
        for name in names:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return None


def _processor_pixel_limit(processor: Any, modality: str, name: str) -> object:
    component = getattr(processor, f"{modality}_processor", None)
    if component is not None:
        size = getattr(component, "size", None)
        if isinstance(size, Mapping):
            size_name = "shortest_edge" if name == "min_pixels" else "longest_edge"
            if size_name in size:
                return size[size_name]
    return _processor_setting(processor, modality, name)


def _selected_video_frame_count(
    media: MediaRef, *, processor: Any, sample_key: str
) -> int:
    total_frames = _positive_int(
        _metadata_value(media, "num_frames", "video_num_frames", "frames"),
        name="num_frames",
        sample_key=sample_key,
    )
    component = getattr(processor, "video_processor", None)
    if component is None:
        raise ValueError("Qwen processor does not expose a video processor.")
    if bool(getattr(component, "do_sample_frames", False)):
        raise ValueError(
            "Qwen Energon packing does not support processors that sample decoded "
            "video frames. Prepare the selected frame count offline or disable "
            "do_sample_frames."
        )

    selected_frames = total_frames
    if isinstance(media.value, str):
        settings = component.to_dict()
        num_frames = settings.get("num_frames")
        fps = settings.get("fps")
        if (
            num_frames is None
            and fps is None
            and settings.get("max_frames") is not None
        ):
            # Keep this in sync with get_multimodal_default_settings_from_processor(),
            # which supplies max_frames to Transformers load_video for path inputs.
            num_frames = settings["max_frames"]
        if num_frames is not None:
            selected_frames = _positive_int(
                num_frames,
                name="video processor num_frames",
                sample_key=sample_key,
            )
        elif fps is not None:
            source_fps = _metadata_value(media, "fps", "video_fps")
            if (
                isinstance(source_fps, bool)
                or not isinstance(source_fps, (int, float))
                or source_fps <= 0
            ):
                raise ValueError(
                    f"Qwen sample {sample_key!r} requires positive video_fps "
                    "metadata for processor fps sampling."
                )
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
                raise ValueError("Qwen video processor fps must be positive.")
            selected_frames = int(total_frames / source_fps * fps)
            if selected_frames <= 0 or selected_frames > total_frames:
                raise ValueError(
                    f"Qwen sample {sample_key!r} video sampling selects "
                    f"{selected_frames} frames from {total_frames}."
                )

    prepared_count = _metadata_value(media, "sampled_num_frames")
    if prepared_count is not None:
        prepared_count = _positive_int(
            prepared_count,
            name="sampled_num_frames",
            sample_key=sample_key,
        )
        if prepared_count != selected_frames:
            raise ValueError(
                f"Qwen sample {sample_key!r} prepared sampled_num_frames="
                f"{prepared_count!r}, but processor settings select {selected_frames}."
            )
    return selected_frames


def _smart_resize(
    *,
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Qwen media aspect ratio must not exceed 200.")

    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


def _predicted_grid(
    media: MediaRef,
    *,
    processor: Any,
    sample_key: str,
) -> tuple[int, int, int]:
    modality = media.modality
    if modality not in {"image", "video"}:
        raise ValueError(
            f"Qwen sample {sample_key!r} has unsupported media modality {modality!r}."
        )

    patch_size = _positive_int(
        _processor_setting(processor, modality, "patch_size"),
        name=f"{modality} processor patch_size",
        sample_key=sample_key,
    )
    merge_size = _positive_int(
        _processor_setting(processor, modality, "merge_size", "spatial_merge_size"),
        name=f"{modality} processor merge_size",
        sample_key=sample_key,
    )
    min_pixels = _positive_int(
        _processor_pixel_limit(processor, modality, "min_pixels"),
        name=f"{modality} processor min_pixels",
        sample_key=sample_key,
    )
    max_pixels = _positive_int(
        _processor_pixel_limit(processor, modality, "max_pixels"),
        name=f"{modality} processor max_pixels",
        sample_key=sample_key,
    )
    if min_pixels > max_pixels:
        raise ValueError("Qwen processor min_pixels must not exceed max_pixels.")

    width = _positive_int(
        _metadata_value(media, "width", f"{modality}_width"),
        name="width",
        sample_key=sample_key,
    )
    height = _positive_int(
        _metadata_value(media, "height", f"{modality}_height"),
        name="height",
        sample_key=sample_key,
    )
    factor = patch_size * merge_size
    resized_height, resized_width = _smart_resize(
        height=height,
        width=width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    grid_t = 1
    if modality == "video":
        num_frames = _selected_video_frame_count(
            media, processor=processor, sample_key=sample_key
        )
        temporal_patch_size = _positive_int(
            _processor_setting(processor, modality, "temporal_patch_size"),
            name="video processor temporal_patch_size",
            sample_key=sample_key,
        )
        grid_t = math.ceil(num_frames / temporal_patch_size)

    return grid_t, resized_height // patch_size, resized_width // patch_size


def _visual_token_count(grids: Sequence[tuple[int, int, int]], merge_size: int) -> int:
    return sum(t * h * w for t, h, w in grids) // (merge_size**2)


def _resolve_media_token_id(processor: Any, modality: str) -> int:
    tokenizer = processor.tokenizer
    for source in (processor, tokenizer):
        value = getattr(source, f"{modality}_token_id", None)
        if isinstance(value, int):
            return value

    token_strings = (
        ("<|image_pad|>", "<image>")
        if modality == "image"
        else ("<|video_pad|>", "<video>")
    )
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        for token in token_strings:
            value = convert(token)
            if isinstance(value, int) and value >= 0 and value != unknown_id:
                return value
    raise ValueError(f"Qwen processor does not expose a {modality} token id.")


def _tokenize_before_media(
    sample: CanonicalSFTSample,
    *,
    adapter: HFMultimodalSFTProcessorAdapter,
) -> list[dict[str, Any]]:
    processor = adapter.processor
    tokenizer = processor.tokenizer
    messages = _normalize_messages(sample)
    tokenized: list[dict[str, Any]] = []
    previous_rendered = ""

    for index, message in enumerate(messages):
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": adapter.add_generation_prompt
            and message["role"] in {"user", "tool"},
            "tokenize": False,
            "add_special_tokens": False,
        }
        if sample.tools is not None:
            template_kwargs["tools"] = sample.tools
        rendered = processor.apply_chat_template(
            messages[: index + 1], **template_kwargs
        )
        if not isinstance(rendered, str):
            raise TypeError("Qwen apply_chat_template must return a string.")
        chunk = rendered[get_first_index_that_differs(previous_rendered, rendered) :]
        if index == 0 and adapter.add_bos and tokenizer.bos_token is not None:
            if not chunk.startswith(tokenizer.bos_token):
                chunk = tokenizer.bos_token + chunk
        if index == len(messages) - 1 and adapter.add_eos:
            eos_token = tokenizer.eos_token
            if eos_token is not None and not chunk.rstrip("\n").endswith(eos_token):
                chunk += eos_token

        result = tokenizer(
            text=chunk,
            return_tensors="pt",
            add_special_tokens=False,
        )
        token_ids = result.get("input_ids")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim not in (1, 2):
            raise ValueError("Qwen tokenizer output must contain tensor input_ids.")
        if token_ids.ndim == 2:
            if token_ids.shape[0] != 1:
                raise ValueError(
                    "Qwen pre-encoding accepts one conversation at a time."
                )
            token_ids = token_ids[0]
        new_message = message.copy()
        new_message["token_ids"] = token_ids.to(dtype=torch.long)
        tokenized.append(new_message)
        previous_rendered = rendered
    return tokenized


def _grid_rows(
    message_log: list[dict[str, Any]], key: str
) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for message in message_log:
        value = message.get(key)
        if value is None:
            continue
        if not isinstance(value, PackedTensor):
            raise TypeError(f"Qwen processor field {key!r} must be a PackedTensor.")
        tensor = value.as_tensor()
        if tensor is None or tensor.ndim != 2 or tensor.shape[1] != 3:
            raise ValueError(f"Qwen processor field {key!r} must have shape [N, 3].")
        rows.extend((int(row[0]), int(row[1]), int(row[2])) for row in tensor.tolist())
    return rows


def _materialize_selected_media(sample: CanonicalSFTSample) -> CanonicalSFTSample:
    media: list[MediaRef] = []
    for ref in sample.media:
        value = ref.value
        if isinstance(value, (bytes, bytearray, memoryview)):
            if ref.modality == "image":
                with Image.open(BytesIO(bytes(value))) as image:
                    value = image.convert("RGB")
            elif ref.modality == "video":
                value = decode_selected_av_bytes(value, modality="video")
        media.append(MediaRef(ref.modality, value, ref.metadata))
    return CanonicalSFTSample.derive_from(sample, media=media)


def _declared_model_inputs(processor: Any) -> set[str]:
    names: set[str] = set()
    for source in (
        processor,
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ):
        if source is not None:
            names.update(getattr(source, "model_input_names", ()))
    return names


@supports_model_families("qwen")
class QwenVLSFTTaskEncoder(GenericSFTTaskEncoder):
    """Pre-compute exact Qwen visual expansion before media processing."""

    decoder = None

    def __init__(
        self,
        *,
        adapter: SFTProcessorAdapter,
        cooker_functions: Sequence[SFTCooker],
        packing_hooks: EnergonPackingHooks[Any, Any, Any] | None,
        include_source_ids: bool,
    ) -> None:
        if not isinstance(adapter, HFMultimodalSFTProcessorAdapter):
            raise TypeError("Qwen SFT requires the Hugging Face processor adapter.")
        super().__init__(
            adapter=adapter,
            cooker_functions=cooker_functions,
            packing_hooks=packing_hooks,
            include_source_ids=include_source_ids,
        )
        self._hf_adapter = adapter

    @stateless
    def preencode_sample(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        message_log = _tokenize_before_media(sample, adapter=self._hf_adapter)
        processor = self._hf_adapter.processor
        predicted_grids = [
            _predicted_grid(media, processor=processor, sample_key=sample.__key__)
            for media in sample.media
        ]

        image_token_id = _resolve_media_token_id(processor, "image")
        video_token_id = _resolve_media_token_id(processor, "video")
        token_ids = torch.cat([message["token_ids"] for message in message_log])
        placeholder_count = int(
            ((token_ids == image_token_id) | (token_ids == video_token_id)).sum().item()
        )
        if placeholder_count != len(sample.media):
            raise ValueError(
                f"Qwen sample {sample.__key__!r} has {placeholder_count} visual "
                f"placeholder tokens for {len(sample.media)} media items."
            )

        merge_sizes = {
            _positive_int(
                _processor_setting(
                    processor, media.modality, "merge_size", "spatial_merge_size"
                ),
                name=f"{media.modality} processor merge_size",
                sample_key=sample.__key__,
            )
            for media in sample.media
        }
        if len(merge_sizes) > 1:
            raise ValueError("Qwen image and video processors need one merge size.")
        merge_size = merge_sizes.pop() if merge_sizes else 1
        packing_cost = (
            len(token_ids)
            - placeholder_count
            + _visual_token_count(predicted_grids, merge_size)
        )
        if packing_cost >= self._hf_adapter.max_sequence_length:
            raise ValueError(
                f"Qwen sample {sample.__key__!r} expanded length {packing_cost} "
                f"reaches max_sequence_length={self._hf_adapter.max_sequence_length}."
            )

        modalities = tuple(media.modality for media in sample.media)
        return EncodedSFTSample.derive_from(
            sample,
            message_log=message_log,
            length=len(token_ids),
            packing_cost=packing_cost,
            loss_multiplier=1.0,
            group_key=(self._hf_adapter.fingerprint, "qwen_vl", modalities),
            sample_key=sample.__key__,
            pending_sample=sample,
        )

    @stateless
    def postencode_sample(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        pending = sample.pending_sample
        if pending is None:
            raise ValueError("Qwen post-encoding requires a pending canonical sample.")
        encoded = self._hf_adapter.encode(_materialize_selected_media(pending))
        processor = self._hf_adapter.processor

        predicted_image_grids = [
            _predicted_grid(media, processor=processor, sample_key=sample.sample_key)
            for media in pending.media
            if media.modality == "image"
        ]
        predicted_video_grids = [
            _predicted_grid(media, processor=processor, sample_key=sample.sample_key)
            for media in pending.media
            if media.modality == "video"
        ]
        actual_image_grids = _grid_rows(encoded.message_log, "image_grid_thw")
        actual_video_grids = _grid_rows(encoded.message_log, "video_grid_thw")
        if actual_image_grids != predicted_image_grids:
            raise ValueError(
                f"Qwen sample {sample.sample_key!r} predicted image grids "
                f"{predicted_image_grids}, but the processor emitted {actual_image_grids}."
            )
        if actual_video_grids != predicted_video_grids:
            raise ValueError(
                f"Qwen sample {sample.sample_key!r} predicted video grids "
                f"{predicted_video_grids}, but the processor emitted {actual_video_grids}."
            )
        if encoded.length != sample.packing_cost:
            raise ValueError(
                f"Qwen sample {sample.sample_key!r} predicted expanded length "
                f"{sample.packing_cost}, but the processor emitted {encoded.length}."
            )

        model_fields = {
            key
            for message in encoded.message_log
            for key in message
            if key != "token_ids"
        }
        required_fields: set[str] = set()
        if predicted_image_grids:
            required_fields.update({"pixel_values", "image_grid_thw"})
        if predicted_video_grids:
            required_fields.update({"pixel_values_videos", "video_grid_thw"})
        declared_fields = _declared_model_inputs(processor)
        if "mm_token_type_ids" in declared_fields:
            required_fields.add("mm_token_type_ids")
        if predicted_video_grids and "second_per_grid_ts" in declared_fields:
            required_fields.add("second_per_grid_ts")
        missing = required_fields - model_fields
        if missing:
            raise ValueError(
                f"Qwen sample {sample.sample_key!r} processor output is missing "
                f"fields {sorted(missing)!r}."
            )
        return EncodedSFTSample.derive_from(
            encoded,
            group_key=sample.group_key,
            pending_sample=None,
        )


__all__ = ["QwenVLSFTTaskEncoder"]
