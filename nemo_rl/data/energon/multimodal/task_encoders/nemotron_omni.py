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
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from nemo_rl.data.energon.multimodal.model_families import supports_model_families
from nemo_rl.data.energon.multimodal.packing import EnergonPackingHooks
from nemo_rl.data.energon.multimodal.task_encoders.base import SFTCooker
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    HFMultimodalSFTProcessorAdapter,
    SFTProcessorAdapter,
    _normalize_messages,
)
from nemo_rl.data.energon.multimodal.task_encoders.media import (
    decode_selected_av_bytes,
)
from nemo_rl.data.energon.multimodal.task_encoders.nemotron_sft import (
    COMPACT_IMAGE_PLACEHOLDER,
    NemotronSFTTaskEncoder,
    NemotronVisualSFTProcessorAdapter,
    _expand_visual_placeholders,
    _normalize_assistant_thinking,
    _predicted_visual_cost,
    _token_id,
    _video_prompt,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    EncodedSFTSample,
    MediaRef,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.data.multimodal_utils import PackedTensor

SOUND_PLACEHOLDER = "<so_embedding>"
SOUND_START = "<so_start>"
SOUND_END = "<so_end>"
_SOUND_MODEL_INPUT_KEYS = ("sound_clips", "sound_length")
_VISUAL_MODEL_INPUT_KEYS = ("imgs_sizes", "num_frames", "pixel_values")

# Audio frame and subsampling math follows Megatron-Bridge revision
# 8c46dc4259080c510b7455f43e836fdff222c5d3,
# models/nemotron_omni/nemotron_omni_utils.py. Per-clip sizing follows the
# Energon reference at revision 6822175d92a40e0528be905aee50f5930cfa0c98,
# examples/multimodal/data_loading/audio_processing.py.


@dataclass(frozen=True)
class _AudioPlan:
    clip_sample_counts: tuple[int, ...]
    valid_frame_counts: tuple[int, ...]
    embedding_counts: tuple[int, ...]

    @property
    def total_embeddings(self) -> int:
        return sum(self.embedding_counts)


def _required_metadata_int(ref: MediaRef, key: str) -> int:
    value = dict(ref.metadata).get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"Nemotron audio media requires positive integer metadata {key!r}; "
            f"got {value!r}. Exact lazy audio sizing requires immutable source "
            "length and sample-rate metadata."
        )
    return value


def _source_audio_samples(ref: MediaRef) -> tuple[int, int]:
    metadata = dict(ref.metadata)
    source_sampling_rate = _required_metadata_int(ref, "audio_sample_rate")
    source_samples = metadata.get("audio_num_samples")
    if source_samples is None:
        duration = metadata.get("audio_duration")
        if type(duration) not in (int, float) or duration <= 0:
            raise ValueError(
                "Nemotron audio media requires positive audio_num_samples or "
                "audio_duration metadata for lazy width prediction."
            )
        source_samples = round(float(duration) * source_sampling_rate)
    if type(source_samples) is not int or source_samples <= 0:
        raise ValueError(
            "Nemotron audio media requires positive integer metadata "
            f"'audio_num_samples'; got {source_samples!r}."
        )
    return source_samples, source_sampling_rate


def _subsampled_length(frame_count: int, subsampling_factor: int) -> int:
    length = frame_count
    for _ in range(int(math.log2(subsampling_factor))):
        # Parakeet uses kernel_size=3, stride=2, and padding=1.
        length = (length + 1) // 2
    return max(1, length)


def _audio_plan(
    ref: MediaRef,
    *,
    target_sampling_rate: int,
    hop_length: int,
    subsampling_factor: int,
    clip_duration_seconds: float,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> _AudioPlan:
    source_samples, source_sampling_rate = _source_audio_samples(ref)
    source_duration = source_samples / source_sampling_rate
    if source_duration > max_duration_seconds:
        raise ValueError(
            f"Nemotron audio duration {source_duration:.3f}s exceeds "
            f"max_audio_duration_seconds={max_duration_seconds}."
        )

    target_samples = round(source_samples * target_sampling_rate / source_sampling_rate)
    min_samples = round(min_duration_seconds * target_sampling_rate)
    target_samples = max(target_samples, min_samples)
    clip_samples = round(clip_duration_seconds * target_sampling_rate)
    num_clips = math.ceil(target_samples / clip_samples)
    remainder = target_samples % clip_samples
    last_clip_samples = clip_samples if remainder == 0 else max(remainder, min_samples)
    clip_sample_counts = (clip_samples,) * (num_clips - 1) + (last_clip_samples,)
    valid_frame_counts = tuple(count // hop_length for count in clip_sample_counts)
    if any(count <= 0 for count in valid_frame_counts):
        raise ValueError(
            "Nemotron audio clip produces no mel frames. Increase "
            "min_audio_duration_seconds or reduce the feature-extractor hop length."
        )
    embedding_counts = tuple(
        _subsampled_length(count, subsampling_factor) for count in valid_frame_counts
    )
    return _AudioPlan(
        clip_sample_counts=clip_sample_counts,
        valid_frame_counts=valid_frame_counts,
        embedding_counts=embedding_counts,
    )


def _render_omni_messages(
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
        rendered_content: list[dict[str, str]] = []
        for part in message["content"]:
            part_type = part["type"]
            if part_type == "text":
                rendered_content.append(
                    {"type": "text", "text": str(part.get("text", ""))}
                )
                continue
            if part_type not in {"audio", "image", "video"}:
                raise ValueError(f"Nemotron Omni does not support {part_type!r} media.")
            if media_index >= len(sample.media):
                raise ValueError(
                    "Nemotron Omni message media exceeds the canonical media list."
                )
            ref = sample.media[media_index]
            if ref.modality != part_type:
                raise ValueError(
                    f"Nemotron Omni media order mismatch: expected {part_type!r}, "
                    f"got {ref.modality!r}."
                )
            if part_type == "image":
                text = COMPACT_IMAGE_PLACEHOLDER
            elif part_type == "video":
                text, _ = _video_prompt(
                    ref,
                    temporal_patch_size=temporal_patch_size,
                )
            else:
                text = f"{SOUND_START}{SOUND_PLACEHOLDER}{SOUND_END}"
            rendered_content.append({"type": "text", "text": text})
            occurrences.append((message_index, ref))
            media_index += 1
        message["content"] = rendered_content
    if media_index != len(sample.media):
        raise ValueError(
            f"Nemotron Omni messages reference {media_index}/{len(sample.media)} "
            "media items."
        )
    return messages, occurrences


def _expand_audio_placeholders(
    message_log: list[dict[str, Any]],
    occurrences: Sequence[tuple[int, MediaRef, _AudioPlan]],
    *,
    sound_token_id: int,
) -> None:
    plans_by_message: defaultdict[int, list[_AudioPlan]] = defaultdict(list)
    for message_index, _, plan in occurrences:
        plans_by_message[message_index].append(plan)

    for message_index, plans in plans_by_message.items():
        token_ids = message_log[message_index].get("token_ids")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError(
                "Nemotron Omni tokenized messages require one-dimensional token_ids."
            )
        placeholder_positions = torch.where(token_ids == sound_token_id)[0].tolist()
        if len(placeholder_positions) != len(plans):
            raise ValueError(
                f"Nemotron Omni message {message_index} has "
                f"{len(placeholder_positions)} compact sound placeholders for "
                f"{len(plans)} audio items."
            )
        pieces: list[torch.Tensor] = []
        start = 0
        for position, plan in zip(placeholder_positions, plans, strict=True):
            pieces.append(token_ids[start:position])
            pieces.append(
                torch.full(
                    (plan.total_embeddings,),
                    sound_token_id,
                    dtype=token_ids.dtype,
                    device=token_ids.device,
                )
            )
            start = position + 1
        pieces.append(token_ids[start:])
        message_log[message_index]["token_ids"] = torch.cat(pieces)


def _native_waveform(
    value: Any,
    *,
    expected_samples: int,
    expected_sampling_rate: int,
) -> torch.Tensor:
    payload = value.get() if callable(getattr(value, "get", None)) else value
    if isinstance(payload, str):
        import soundfile as sf

        payload, sampling_rate = sf.read(
            payload,
            dtype="float32",
            always_2d=False,
        )
        if sampling_rate != expected_sampling_rate:
            raise ValueError(
                f"Nemotron audio file sampling rate {sampling_rate} does not match "
                f"immutable audio_sample_rate={expected_sampling_rate}."
            )
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = decode_selected_av_bytes(payload, modality="audio")
    if callable(getattr(payload, "get_audio", None)):
        payload = payload.get_audio().audio_clips
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        tensors = [torch.as_tensor(clip) for clip in payload]
        if not tensors:
            raise ValueError("Nemotron audio payload has no clips.")
        payload = torch.cat(tensors, dim=-1)
    waveform = torch.as_tensor(payload)
    if waveform.ndim == 2:
        waveform = waveform.to(dtype=torch.float32)
        if waveform.shape[-1] == expected_samples:
            waveform = waveform.mean(dim=0)
        elif waveform.shape[0] == expected_samples:
            waveform = waveform.mean(dim=1)
        else:
            raise ValueError(
                "Nemotron audio payload has no axis matching immutable "
                f"audio_num_samples={expected_samples}; got shape "
                f"{tuple(waveform.shape)}."
            )
    elif waveform.ndim != 1:
        raise ValueError(
            f"Nemotron audio payload must be one- or two-dimensional, got "
            f"shape {tuple(waveform.shape)}."
        )
    if waveform.dtype == torch.int16:
        waveform = waveform.to(dtype=torch.float32) / 32768.0
    elif waveform.dtype == torch.int32:
        waveform = waveform.to(dtype=torch.float32) / 2147483648.0
    else:
        waveform = waveform.to(dtype=torch.float32)
    return waveform


def _feature_output(features: Any) -> Mapping[str, Any]:
    if isinstance(features, Mapping):
        return features
    data = getattr(features, "data", None)
    if isinstance(data, Mapping):
        return data
    raise TypeError("Nemotron audio feature extractor must return a mapping.")


class NemotronOmniSFTProcessorAdapter(NemotronVisualSFTProcessorAdapter):
    """Predict Omni image and sound width before loading media payloads."""

    def __init__(
        self,
        *,
        processor: Any,
        max_sequence_length: int,
        patch_dim: int = 16,
        temporal_patch_size: int = 2,
        prompt_format: str = "nemotron-h-5p5-reasoning",
        thinking_trace_format: str = "default",
        audio_subsampling_factor: int | None = None,
        audio_num_mel_bins: int = 128,
        audio_clip_duration_seconds: float = 60.0,
        min_audio_duration_seconds: float = 0.1,
        max_audio_duration_seconds: float = 1800.0,
        add_bos: bool = False,
        add_eos: bool = False,
        add_generation_prompt: bool = False,
    ) -> None:
        super().__init__(
            processor=processor,
            max_sequence_length=max_sequence_length,
            patch_dim=patch_dim,
            temporal_patch_size=temporal_patch_size,
            prompt_format=prompt_format,
            thinking_trace_format=thinking_trace_format,
            add_bos=add_bos,
            add_eos=add_eos,
            add_generation_prompt=add_generation_prompt,
        )
        feature_extractor = getattr(processor, "feature_extractor", None)
        target_sampling_rate = getattr(feature_extractor, "sampling_rate", None)
        if target_sampling_rate is None:
            target_sampling_rate = getattr(processor, "audio_sampling_rate", None)
        hop_length = getattr(feature_extractor, "hop_length", None)
        if hop_length is None:
            hop_length = getattr(processor, "audio_hop_length", None)
        if (
            type(target_sampling_rate) is not int
            or target_sampling_rate <= 0
            or type(hop_length) is not int
            or hop_length <= 0
        ):
            raise ValueError(
                "Nemotron Omni processor feature_extractor must expose positive "
                "integer sampling_rate and hop_length values so pre-encode can "
                "predict audio width without reading the payload."
            )
        if feature_extractor is None:
            from transformers import ParakeetFeatureExtractor

            if type(audio_num_mel_bins) is not int or audio_num_mel_bins <= 0:
                raise ValueError(
                    "audio_num_mel_bins must be a positive integer when the "
                    "processor feature_extractor is absent."
                )
            feature_extractor = ParakeetFeatureExtractor(
                feature_size=audio_num_mel_bins,
                sampling_rate=target_sampling_rate,
                hop_length=hop_length,
            )
        if audio_subsampling_factor is None:
            audio_subsampling_factor = getattr(
                processor,
                "audio_subsampling_factor",
                None,
            )
        if type(audio_subsampling_factor) is not int:
            raise ValueError(
                "Nemotron Omni processor must expose integer audio_subsampling_factor."
            )
        if audio_subsampling_factor <= 0 or audio_subsampling_factor & (
            audio_subsampling_factor - 1
        ):
            raise ValueError("audio_subsampling_factor must be a power of two.")
        if not 0 < min_audio_duration_seconds <= audio_clip_duration_seconds:
            raise ValueError(
                "min_audio_duration_seconds must be positive and no larger than "
                "audio_clip_duration_seconds."
            )
        if max_audio_duration_seconds < audio_clip_duration_seconds:
            raise ValueError(
                "max_audio_duration_seconds must be no smaller than "
                "audio_clip_duration_seconds."
            )
        self.feature_extractor = feature_extractor
        self.target_sampling_rate = target_sampling_rate
        self.hop_length = hop_length
        self.audio_subsampling_factor = audio_subsampling_factor
        self.audio_clip_duration_seconds = audio_clip_duration_seconds
        self.min_audio_duration_seconds = min_audio_duration_seconds
        self.max_audio_duration_seconds = max_audio_duration_seconds
        fingerprint_data = {
            "visual_fingerprint": self.fingerprint,
            "target_sampling_rate": target_sampling_rate,
            "hop_length": hop_length,
            "audio_subsampling_factor": audio_subsampling_factor,
            "audio_num_mel_bins": audio_num_mel_bins,
            "audio_clip_duration_seconds": audio_clip_duration_seconds,
            "min_audio_duration_seconds": min_audio_duration_seconds,
            "max_audio_duration_seconds": max_audio_duration_seconds,
        }
        encoded = json.dumps(fingerprint_data, sort_keys=True).encode()
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    def _plan_audio(self, ref: MediaRef) -> _AudioPlan:
        return _audio_plan(
            ref,
            target_sampling_rate=self.target_sampling_rate,
            hop_length=self.hop_length,
            subsampling_factor=self.audio_subsampling_factor,
            clip_duration_seconds=self.audio_clip_duration_seconds,
            min_duration_seconds=self.min_audio_duration_seconds,
            max_duration_seconds=self.max_audio_duration_seconds,
        )

    def preencode(self, sample: CanonicalSFTSample) -> EncodedSFTSample:
        """Expand sound placeholders and predict visual width from metadata."""
        messages, occurrences = _render_omni_messages(
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
        audio_occurrences = [
            (message_index, ref, self._plan_audio(ref))
            for message_index, ref in occurrences
            if ref.modality == "audio"
        ]
        sound_token_id = _token_id(self.processor.tokenizer, SOUND_PLACEHOLDER)
        _expand_audio_placeholders(
            message_log,
            audio_occurrences,
            sound_token_id=sound_token_id,
        )

        length = sum(len(message["token_ids"]) for message in message_log)
        visual_embeddings = 0
        visual_placeholders = 0
        for _, ref in occurrences:
            if ref.modality == "audio":
                continue
            embeddings, placeholders = _predicted_visual_cost(
                ref,
                patch_dim=self.patch_dim,
                temporal_patch_size=self.temporal_patch_size,
            )
            visual_embeddings += embeddings
            visual_placeholders += placeholders
        audio_embeddings = sum(
            plan.total_embeddings for _, _, plan in audio_occurrences
        )
        packing_cost = length + visual_embeddings - visual_placeholders
        if packing_cost > self.max_sequence_length:
            raise ValueError(
                f"Nemotron Omni sample {sample.__key__!r} has expanded length "
                f"{packing_cost}, above max_sequence_length="
                f"{self.max_sequence_length}."
            )
        model_input_keys = (
            *(_VISUAL_MODEL_INPUT_KEYS if visual_placeholders else ()),
        ) + (*(_SOUND_MODEL_INPUT_KEYS if audio_occurrences else ()),)
        media_embeddings = visual_embeddings + audio_embeddings
        cost_bucket = (
            0 if media_embeddings <= 256 else 1 if media_embeddings <= 2_048 else 2
        )
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

    def _process_audio(
        self,
        ref: MediaRef,
    ) -> tuple[dict[str, PackedTensor], _AudioPlan]:
        plan = self._plan_audio(ref)
        source_samples, source_sampling_rate = _source_audio_samples(ref)
        waveform = _native_waveform(
            ref.value,
            expected_samples=source_samples,
            expected_sampling_rate=source_sampling_rate,
        )
        if len(waveform) != source_samples:
            raise ValueError(
                f"Nemotron audio payload has {len(waveform)} samples, but immutable "
                f"metadata predicted {source_samples}."
            )
        target_samples = round(
            source_samples * self.target_sampling_rate / source_sampling_rate
        )
        target_samples = max(
            target_samples,
            round(self.min_audio_duration_seconds * self.target_sampling_rate),
        )
        if source_sampling_rate != self.target_sampling_rate:
            # Librosa is an optional audio dependency, so keep it off non-audio
            # import paths. ParakeetFeatureExtractor also requires this package.
            import librosa

            waveform = torch.from_numpy(
                librosa.resample(
                    waveform.numpy(),
                    orig_sr=source_sampling_rate,
                    target_sr=self.target_sampling_rate,
                )
            )
        if len(waveform) < target_samples:
            waveform = F.pad(waveform, (0, target_samples - len(waveform)))
        elif len(waveform) > target_samples:
            waveform = waveform[:target_samples]
        planned_samples = sum(plan.clip_sample_counts)
        if len(waveform) < planned_samples:
            waveform = F.pad(waveform, (0, planned_samples - len(waveform)))

        clip_width = round(self.audio_clip_duration_seconds * self.target_sampling_rate)
        clips = list(torch.split(waveform, clip_width))
        if tuple(len(clip) for clip in clips) != plan.clip_sample_counts:
            raise ValueError(
                "Nemotron audio clip sizes changed after payload processing: "
                f"predicted {plan.clip_sample_counts}, got "
                f"{tuple(len(clip) for clip in clips)}."
            )

        mel_features: list[torch.Tensor] = []
        valid_lengths: list[int] = []
        actual_embedding_counts: list[int] = []
        for clip in clips:
            features = _feature_output(
                self.feature_extractor(
                    clip,
                    sampling_rate=self.target_sampling_rate,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            )
            mel = torch.as_tensor(features.get("input_features"))
            attention_mask = torch.as_tensor(features.get("attention_mask"))
            if mel.ndim != 3 or mel.shape[0] != 1:
                raise ValueError(
                    "Nemotron audio feature extractor must return input_features "
                    f"with shape [1, frames, mel_bins], got {tuple(mel.shape)}."
                )
            if attention_mask.shape != mel.shape[:2]:
                raise ValueError(
                    "Nemotron audio attention_mask must match the feature frame "
                    f"shape, got {tuple(attention_mask.shape)} for "
                    f"{tuple(mel.shape)}."
                )
            mask = attention_mask.to(dtype=torch.bool)
            valid_length = int(mask.sum().item())
            expected_mask = torch.arange(mel.shape[1])[None, :] < valid_length
            if valid_length <= 0 or not torch.equal(mask.cpu(), expected_mask):
                raise ValueError(
                    "Nemotron audio attention_mask must contain one non-empty "
                    "contiguous prefix."
                )
            mel_features.append(mel[0])
            valid_lengths.append(valid_length)
            actual_embedding_counts.append(
                _subsampled_length(valid_length, self.audio_subsampling_factor)
            )

        if tuple(valid_lengths) != plan.valid_frame_counts:
            raise ValueError(
                "Nemotron audio frame count changed after feature extraction: "
                f"predicted {plan.valid_frame_counts}, got {tuple(valid_lengths)}."
            )
        if tuple(actual_embedding_counts) != plan.embedding_counts:
            raise ValueError(
                "Nemotron audio embedding count changed after feature extraction: "
                f"predicted {plan.embedding_counts}, got "
                f"{tuple(actual_embedding_counts)}."
            )
        max_frames = max(mel.shape[0] for mel in mel_features)
        num_mel_bins = {int(mel.shape[1]) for mel in mel_features}
        if len(num_mel_bins) != 1:
            raise ValueError("Nemotron audio clips must use one mel feature width.")
        sound_clips = mel_features[0].new_zeros(
            (len(mel_features), max_frames, num_mel_bins.pop())
        )
        for clip_index, mel in enumerate(mel_features):
            sound_clips[clip_index, : mel.shape[0]] = mel
        return {
            "sound_clips": PackedTensor(
                sound_clips,
                dim_to_pack=0,
                pad_to_max_shape=True,
            ),
            "sound_length": PackedTensor(
                torch.tensor(valid_lengths, dtype=torch.long),
                dim_to_pack=0,
            ),
        }, plan

    def postencode(self, sample: EncodedSFTSample) -> EncodedSFTSample:
        """Load selected Omni media and verify the predicted expanded width."""
        pending_sample = sample.pending_sample
        if pending_sample is None:
            raise ValueError(
                f"Nemotron Omni sample {sample.sample_key!r} has no pending media."
            )
        _, occurrences = _render_omni_messages(
            pending_sample,
            temporal_patch_size=self.temporal_patch_size,
            prompt_format=self.prompt_format,
            thinking_trace_format=self.thinking_trace_format,
        )
        message_log = deepcopy(sample.message_log)
        inputs_by_message: defaultdict[int, defaultdict[str, list[PackedTensor]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        actual_visual_embeddings = 0
        visual_placeholders = 0
        visual_occurrences: list[tuple[int, tuple[int, ...]]] = []
        actual_audio_embeddings = 0
        for message_index, ref in occurrences:
            if ref.modality == "audio":
                media_inputs, plan = self._process_audio(ref)
                actual_audio_embeddings += plan.total_embeddings
            else:
                media_inputs, embeddings = self._process_media(ref)
                actual_visual_embeddings += embeddings
                _, placeholders = _predicted_visual_cost(
                    ref,
                    patch_dim=self.patch_dim,
                    temporal_patch_size=self.temporal_patch_size,
                )
                visual_placeholders += placeholders
                if embeddings % placeholders:
                    raise ValueError(
                        f"Nemotron {ref.modality} produced {embeddings} visual "
                        f"features for {placeholders} placeholder rows."
                    )
                visual_occurrences.append(
                    (message_index, (embeddings // placeholders,) * placeholders)
                )
            for key, value in media_inputs.items():
                inputs_by_message[message_index][key].append(value)

        sound_token_id = _token_id(self.processor.tokenizer, SOUND_PLACEHOLDER)
        actual_sound_placeholders = sum(
            int((message["token_ids"] == sound_token_id).sum().item())
            for message in message_log
        )
        if actual_sound_placeholders != actual_audio_embeddings:
            raise ValueError(
                f"Nemotron Omni sound placeholder count changed: found "
                f"{actual_sound_placeholders}, expected {actual_audio_embeddings}."
            )
        actual_cost = sample.length + actual_visual_embeddings - visual_placeholders
        if actual_cost != sample.packing_cost:
            raise ValueError(
                f"Nemotron Omni sample {sample.sample_key!r} expanded length "
                f"changed after media processing: predicted {sample.packing_cost}, "
                f"got {actual_cost}."
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
                f"Nemotron Omni sample {sample.sample_key!r} expanded to "
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
class NemotronOmniSFTTaskEncoder(NemotronSFTTaskEncoder):
    """Run the split Nemotron Omni visual and sound encoding lifecycle."""

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
        audio_subsampling_factor: int | None = None,
        audio_num_mel_bins: int = 128,
        audio_clip_duration_seconds: float = 60.0,
        min_audio_duration_seconds: float = 0.1,
        max_audio_duration_seconds: float = 1800.0,
    ) -> None:
        if isinstance(adapter, NemotronOmniSFTProcessorAdapter):
            omni_adapter = adapter
        elif isinstance(adapter, HFMultimodalSFTProcessorAdapter):
            omni_adapter = NemotronOmniSFTProcessorAdapter(
                processor=adapter.processor,
                max_sequence_length=adapter.max_sequence_length,
                patch_dim=patch_dim,
                temporal_patch_size=temporal_patch_size,
                prompt_format=prompt_format,
                thinking_trace_format=thinking_trace_format,
                audio_subsampling_factor=audio_subsampling_factor,
                audio_num_mel_bins=audio_num_mel_bins,
                audio_clip_duration_seconds=audio_clip_duration_seconds,
                min_audio_duration_seconds=min_audio_duration_seconds,
                max_audio_duration_seconds=max_audio_duration_seconds,
                add_bos=adapter.add_bos,
                add_eos=adapter.add_eos,
                add_generation_prompt=adapter.add_generation_prompt,
            )
        else:
            raise TypeError(
                "Nemotron Omni SFT requires the Hugging Face or Nemotron Omni "
                "processor adapter."
            )
        super().__init__(
            adapter=omni_adapter,
            cooker_functions=cooker_functions,
            packing_hooks=packing_hooks,
            include_source_ids=include_source_ids,
            patch_dim=patch_dim,
            temporal_patch_size=temporal_patch_size,
            prompt_format=prompt_format,
            thinking_trace_format=thinking_trace_format,
        )


__all__ = [
    "NemotronOmniSFTProcessorAdapter",
    "NemotronOmniSFTTaskEncoder",
    "SOUND_END",
    "SOUND_PLACEHOLDER",
    "SOUND_START",
]
