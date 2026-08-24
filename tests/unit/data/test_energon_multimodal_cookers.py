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

from dataclasses import FrozenInstanceError

import pytest

from nemo_rl.data.energon.multimodal.cookers.generic import cook_conversation
from nemo_rl.data.energon.multimodal.cookers.nemotron import (
    GRANARY_ENGLISH_PROMPT,
    cook_general_conversations_jsonl,
    cook_general_conversations_webdataset,
    cook_granary_english_jsonl,
    cook_granary_english_webdataset,
)
from nemo_rl.data.energon.multimodal.model_families import (
    get_supported_model_families,
)


def _sample(payload, **members):
    return {
        "__key__": "sample-0",
        "__restore_key__": ("sample-0",),
        "__subflavors__": {},
        "json": payload,
        **members,
    }


def test_generic_cooker_freezes_explicit_media_metadata_without_opening_media():
    media_value = object()
    cooked = cook_conversation(
        _sample(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "media_index": 0}],
                    }
                ],
                "media": [
                    {
                        "type": "image",
                        "value": media_value,
                        "metadata": {"width": 32, "height": 16},
                    }
                ],
            }
        )
    )

    assert cooked.media[0].value is media_value
    assert cooked.media[0].metadata == (("height", 16), ("width", 32))
    with pytest.raises(FrozenInstanceError):
        cooked.media[0].metadata = ()


def test_general_wds_cooker_preserves_aliases_order_and_metadata():
    image, video, speech = object(), object(), object()
    cooked = cook_general_conversations_webdataset(
        _sample(
            {
                "images": {
                    "member": "sample-0.picture.png",
                    "metadata": {"width": 64, "height": 32},
                },
                "video-sound": {
                    "member": "sample-0.clip.mp4",
                    "metadata": {"video_num_frames": 8, "video_fps": 2.0},
                },
                "speech": {
                    "member": "sample-0.voice.flac",
                    "metadata": {"audio_duration": 1.5, "audio_sample_rate": 16000},
                },
                "conversations": [
                    {
                        "from": "human",
                        "value": "See <images>, watch <video-sound>, hear <speech>.",
                    },
                    {"from": "gpt", "value": "Done."},
                ],
            },
            png=image,
            mp4=video,
            flac=speech,
        )
    )

    assert [message["role"] for message in cooked.messages] == ["user", "assistant"]
    assert [ref.modality for ref in cooked.media] == [
        "image",
        "video",
        "audio",
        "audio",
    ]
    assert [ref.value for ref in cooked.media] == [image, video, video, speech]
    assert cooked.media[0].metadata == (("height", 32), ("width", 64))
    media_indexes = [
        part["media_index"]
        for part in cooked.messages[0]["content"]
        if "media_index" in part
    ]
    assert media_indexes == [
        0,
        1,
        2,
        3,
    ]


def test_general_jsonl_cooker_keeps_lazy_paths_and_rejects_source_errors():
    cooked = cook_general_conversations_jsonl(
        _sample(
            {
                "audio": {
                    "path": "/media/voice.wav",
                    "metadata": {"audio_duration": 2.0},
                },
                "conversations": [
                    {"from": "user", "value": "<audio>"},
                    {"from": "assistant", "value": "Transcript"},
                ],
            }
        )
    )
    assert cooked.media[0].value == "/media/voice.wav"
    assert cooked.media[0].metadata == (("audio_duration", 2.0),)
    assert cooked.messages[0]["content"] == [
        {"type": "audio", "media_index": 0},
        {"type": "text", "text": " "},
    ]

    with pytest.raises(ValueError, match="Retrieved 0/1 image"):
        cook_general_conversations_jsonl(
            _sample(
                {
                    "image": "unused.jpg",
                    "conversations": [{"from": "user", "value": "text only"}],
                }
            )
        )
    with pytest.raises(ValueError, match="has no 'image' media field"):
        cook_general_conversations_jsonl(
            _sample({"conversations": [{"from": "user", "value": "<image>"}]})
        )


@pytest.mark.parametrize(
    ("cooker", "payload", "members", "expected_value"),
    [
        (
            cook_granary_english_webdataset,
            {
                "audio_filepath": {
                    "member": "sample-0.audio.flac",
                    "metadata": {
                        "audio_num_samples": 32000,
                        "audio_sample_rate": 16000,
                    },
                },
                "text": "Hello world.",
            },
            {"flac": "decoded-later"},
            "decoded-later",
        ),
        (
            cook_granary_english_jsonl,
            {
                "audio_filepath": {
                    "path": "/media/audio.flac",
                    "metadata": {
                        "audio_num_samples": 32000,
                        "audio_sample_rate": 16000,
                    },
                },
                "text": "Hello world.",
            },
            {},
            "/media/audio.flac",
        ),
    ],
)
def test_granary_cookers_build_the_fixed_asr_conversation(
    cooker, payload, members, expected_value
):
    cooked = cooker(_sample(payload, **members))

    assert cooked.media[0].modality == "audio"
    assert cooked.media[0].value == expected_value
    assert cooked.media[0].metadata == (
        ("audio_num_samples", 32000),
        ("audio_sample_rate", 16000),
    )
    assert cooked.messages == [
        {
            "role": "user",
            "content": [
                {"type": "audio", "media_index": 0},
                {
                    "type": "text",
                    "text": GRANARY_ENGLISH_PROMPT.removeprefix("<audio>"),
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world."}]},
    ]


def test_new_cookers_declare_nemotron_support():
    for cooker in (
        cook_general_conversations_jsonl,
        cook_general_conversations_webdataset,
        cook_granary_english_jsonl,
        cook_granary_english_webdataset,
    ):
        assert get_supported_model_families(cooker) == frozenset({"nemotron"})
