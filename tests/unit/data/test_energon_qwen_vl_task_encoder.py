from copy import deepcopy
import io

import pytest
import torch
from PIL import Image

from nemo_rl.data.energon.multimodal.model_families import (
    get_supported_model_families,
)
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    HFMultimodalSFTProcessorAdapter,
)
from nemo_rl.data.energon.multimodal.task_encoders.qwen_vl import (
    QwenVLSFTTaskEncoder,
    _predicted_grid,
)
from nemo_rl.data.energon.multimodal.types import (
    CanonicalSFTSample,
    MediaRef,
    freeze_media_metadata,
)
from nemo_rl.data.multimodal_utils import PackedTensor

_IMAGE_TOKEN_ID = 101
_VIDEO_TOKEN_ID = 102


class _FakeTokenizer:
    bos_token = None
    eos_token = None
    pad_token_id = 0
    image_token_id = _IMAGE_TOKEN_ID
    video_token_id = _VIDEO_TOKEN_ID
    model_input_names = ["input_ids", "attention_mask"]
    name_or_path = "fake-qwen-tokenizer"
    chat_template = "fake-qwen-template"

    def __call__(self, *, text, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        ids: list[int] = []
        position = 0
        markers = {
            "<|image_pad|>": _IMAGE_TOKEN_ID,
            "<|video_pad|>": _VIDEO_TOKEN_ID,
        }
        while position < len(text):
            for marker, token_id in markers.items():
                if text.startswith(marker, position):
                    ids.append(token_id)
                    position += len(marker)
                    break
            else:
                ids.append(ord(text[position]) % 89 + 10)
                position += 1
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _FakeVisionProcessor:
    model_input_names = [
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    ]
    patch_size = 14
    merge_size = 2
    temporal_patch_size = 2
    min_pixels = 28 * 28
    max_pixels = 112 * 112
    size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    do_sample_frames = False

    def to_dict(self):
        return {}


class _FakeQwenProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeVisionProcessor()
    video_processor = _FakeVisionProcessor()
    image_token_id = _IMAGE_TOKEN_ID
    video_token_id = _VIDEO_TOKEN_ID
    model_input_names = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
        "mm_token_type_ids",
    ]
    name_or_path = "fake-qwen-vl"
    chat_template = "fake-qwen-template"
    bos_token = None
    eos_token = None

    def __init__(
        self,
        *,
        alter_video_grid: bool = False,
        omitted_fields: frozenset[str] = frozenset(),
    ):
        self.process_calls = 0
        self.alter_video_grid = alter_video_grid
        self.omitted_fields = omitted_fields
        self.media_values: list[float] = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False

        def render_content(content):
            if isinstance(content, str):
                return content
            rendered = ""
            for part in content:
                if part["type"] == "text":
                    rendered += part["text"]
                elif part["type"] == "image":
                    rendered += "<|image_pad|>"
                elif part["type"] == "video":
                    rendered += "<|video_pad|>"
            return rendered

        return "".join(
            f"<{message['role']}>{render_content(message.get('content', ''))}"
            for message in messages
        )

    def __call__(
        self,
        *,
        text,
        images=None,
        videos=None,
        return_tensors,
        add_special_tokens,
    ):
        self.process_calls += 1
        text = text[0]
        input_ids = self.tokenizer(
            text=text,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )["input_ids"]
        expanded: list[int] = []
        for token_id in input_ids[0].tolist():
            if token_id == _IMAGE_TOKEN_ID:
                expanded.extend([token_id] * 2)
            elif token_id == _VIDEO_TOKEN_ID:
                expanded.extend([token_id] * 2)
            else:
                expanded.append(token_id)
        input_ids = torch.tensor([expanded], dtype=torch.long)
        processed = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.zeros_like(input_ids),
        }
        if images:
            self.media_values.extend(
                float(image.item())
                if isinstance(image, torch.Tensor)
                else float(image.getpixel((0, 0))[0])
                for image in images
            )
            processed["pixel_values"] = torch.cat(
                [torch.full((8, 3), float(image.item())) for image in images]
            )
            processed["image_grid_thw"] = torch.tensor(
                [[1, 2, 4]] * len(images), dtype=torch.long
            )
        if videos:
            self.media_values.extend(float(video.item()) for video in videos)
            processed["pixel_values_videos"] = torch.cat(
                [torch.full((8, 3), float(video.item())) for video in videos]
            )
            video_grid = [3, 2, 2] if self.alter_video_grid else [2, 2, 2]
            processed["video_grid_thw"] = torch.tensor(
                [video_grid] * len(videos), dtype=torch.long
            )
            processed["second_per_grid_ts"] = torch.tensor(
                [0.5] * len(videos), dtype=torch.float32
            )
        for field in self.omitted_fields:
            processed.pop(field, None)
        return processed


def _sample(*, include_metadata: bool = True) -> CanonicalSFTSample:
    image_metadata = {"width": 56, "height": 28} if include_metadata else None
    video_metadata = (
        {"video_width": 28, "video_height": 28, "video_num_frames": 4}
        if include_metadata
        else None
    )
    return CanonicalSFTSample(
        __key__="qwen-0",
        __restore_key__=("qwen-0",),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "media_index": 0},
                    {"type": "text", "text": " then "},
                    {"type": "video", "media_index": 1},
                ],
            },
            {"role": "assistant", "content": "done"},
        ],
        media=[
            MediaRef(
                "image",
                torch.tensor(1.0),
                freeze_media_metadata(image_metadata),
            ),
            MediaRef(
                "video",
                torch.tensor(2.0),
                freeze_media_metadata(video_metadata),
            ),
        ],
        tools=None,
    )


def _encoder(processor: _FakeQwenProcessor) -> QwenVLSFTTaskEncoder:
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=1024,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=False,
    )
    return QwenVLSFTTaskEncoder(
        adapter=adapter,
        cooker_functions=[],
        packing_hooks=None,
        include_source_ids=True,
    )


def test_qwen_preencode_predicts_exact_cost_without_processing_media():
    processor = _FakeQwenProcessor()
    encoder = _encoder(processor)

    preencoded = encoder.preencode_sample(_sample())

    assert processor.process_calls == 0
    assert preencoded.pending_sample is not None
    assert preencoded.packing_cost == preencoded.length + 2
    assert preencoded.group_key[1:] == ("qwen_vl", ("image", "video"))
    assert get_supported_model_families(QwenVLSFTTaskEncoder) == frozenset({"qwen"})


def test_qwen_postencode_preserves_visual_fields_order_and_expanded_cost():
    processor = _FakeQwenProcessor()
    encoder = _encoder(processor)
    preencoded = encoder.preencode_sample(_sample())

    encoded = encoder.postencode_sample(preencoded)
    batch = encoder.batch([encoded])

    assert processor.process_calls == 1
    assert processor.media_values == [1.0, 2.0]
    assert encoded.pending_sample is None
    assert encoded.length == preencoded.packing_cost
    assert batch["message_log"][0] is encoded.message_log
    fields = {
        key: value for message in encoded.message_log for key, value in message.items()
    }
    for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    ):
        assert isinstance(fields[key], PackedTensor)
    assert isinstance(fields["mm_token_type_ids"], torch.Tensor)
    assert (
        fields["mm_token_type_ids"].shape[0]
        == encoded.message_log[0]["token_ids"].shape[0]
    )


def test_qwen_preencode_rejects_missing_lazy_grid_metadata():
    processor = _FakeQwenProcessor()
    encoder = _encoder(processor)

    with pytest.raises(ValueError, match="requires positive integer media metadata"):
        encoder.preencode_sample(_sample(include_metadata=False))
    assert processor.process_calls == 0


def test_qwen_postencode_rejects_processor_grid_mismatch():
    processor = _FakeQwenProcessor(alter_video_grid=True)
    encoder = _encoder(processor)
    preencoded = encoder.preencode_sample(_sample())

    with pytest.raises(ValueError, match="predicted video grids"):
        encoder.postencode_sample(preencoded)


def test_qwen_preencode_rejects_media_reordering():
    processor = _FakeQwenProcessor()
    encoder = _encoder(processor)
    sample = deepcopy(_sample())
    sample.messages[0]["content"][0]["media_index"] = 1
    sample.messages[0]["content"][2]["media_index"] = 0

    with pytest.raises(ValueError, match="referenced once in order"):
        encoder.preencode_sample(sample)


def test_qwen_uses_modality_specific_resize_limits():
    processor = _FakeQwenProcessor()
    processor.image_processor = _FakeVisionProcessor()
    processor.video_processor = _FakeVisionProcessor()
    processor.image_processor.size = {
        "shortest_edge": 28 * 28,
        "longest_edge": 56 * 56,
    }
    processor.video_processor.size = {
        "shortest_edge": 28 * 28,
        "longest_edge": 112 * 112,
    }
    metadata = freeze_media_metadata({"width": 100, "height": 100})

    image_grid = _predicted_grid(
        MediaRef("image", torch.tensor(1.0), metadata),
        processor=processor,
        sample_key="resize",
    )
    video_grid = _predicted_grid(
        MediaRef(
            "video",
            torch.tensor(2.0),
            freeze_media_metadata({"width": 100, "height": 100, "num_frames": 4}),
        ),
        processor=processor,
        sample_key="resize",
    )

    assert image_grid == (1, 4, 4)
    assert video_grid == (2, 8, 8)


@pytest.mark.parametrize("field", ["second_per_grid_ts", "mm_token_type_ids"])
def test_qwen_requires_declared_processor_fields(field: str):
    processor = _FakeQwenProcessor(omitted_fields=frozenset({field}))
    encoder = _encoder(processor)
    preencoded = encoder.preencode_sample(_sample())

    with pytest.raises(ValueError, match=field):
        encoder.postencode_sample(preencoded)


def test_qwen_decodes_selected_raw_image_bytes_only_during_postencode():
    buffer = io.BytesIO()
    Image.new("RGB", (56, 28), color=(7, 0, 0)).save(buffer, format="PNG")
    sample = _sample()
    sample = CanonicalSFTSample.derive_from(
        sample,
        media=[
            MediaRef("image", buffer.getvalue(), sample.media[0].metadata),
            sample.media[1],
        ],
    )
    processor = _FakeQwenProcessor()
    encoder = _encoder(processor)

    preencoded = encoder.preencode_sample(sample)
    assert processor.process_calls == 0

    encoder.postencode_sample(preencoded)
    assert processor.process_calls == 1
    assert processor.media_values == [7.0, 2.0]


def test_qwen_rejects_processor_side_video_frame_sampling():
    processor = _FakeQwenProcessor()
    processor.video_processor = _FakeVisionProcessor()
    processor.video_processor.do_sample_frames = True

    with pytest.raises(ValueError, match="does not support processors that sample"):
        _encoder(processor).preencode_sample(_sample())


def test_qwen_path_video_cost_uses_loader_frame_limit():
    processor = _FakeQwenProcessor()
    processor.video_processor = _FakeVisionProcessor()
    processor.video_processor.to_dict = lambda: {
        "fps": None,
        "num_frames": None,
        "max_frames": 8,
    }
    media = MediaRef(
        "video",
        "clip.mp4",
        freeze_media_metadata({"width": 28, "height": 28, "num_frames": 100}),
    )

    assert _predicted_grid(media, processor=processor, sample_key="path") == (
        4,
        2,
        2,
    )
