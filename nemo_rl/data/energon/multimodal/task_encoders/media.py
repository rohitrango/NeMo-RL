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

from io import BytesIO
from typing import Any, Literal


def decode_selected_av_bytes(
    value: bytes | bytearray | memoryview,
    *,
    modality: Literal["audio", "video"],
) -> Any:
    """Decode one selected raw AV member after packing has selected its row."""
    from megatron.energon.av import AVDecoder

    decoder = AVDecoder(BytesIO(bytes(value)))
    if modality == "audio":
        clips = decoder.get_audio().audio_clips
        if not clips:
            raise ValueError("Selected audio member must decode to at least one clip.")
        return clips

    clips = decoder.get_video().video_clips
    if len(clips) != 1:
        raise ValueError(
            f"Selected {modality} member must decode to one clip; got {len(clips)}."
        )
    return clips[0]


__all__ = ["decode_selected_av_bytes"]
