import io
from copy import deepcopy

import pytest
import torch

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonSourceConfig
from nemo_rl.data.energon.sft import (
    CanonicalSFTSample,
    EnergonSFTTaskEncoder,
    HFMultimodalSFTProcessorAdapter,
    MediaRef,
)
from nemo_rl.data.energon.sft_dataloader import EnergonSFTDataLoader
from nemo_rl.data.multimodal_utils import PackedTensor


class _FakeTokenizer:
    pad_token_id = 0
    model_input_names = ["input_ids", "attention_mask"]
    name_or_path = "fake-tokenizer"
    chat_template = "fake-template"


class _FakeImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw"]


class _FakeQwenProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeImageProcessor()
    model_input_names = [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    ]
    name_or_path = "fake-qwen3-vl"

    def __init__(self, assistant_mask=None):
        self.assistant_mask = assistant_mask or [0, 0, 1, 1]
        self.messages = None
        self.tools = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = deepcopy(messages)
        self.tools = kwargs.get("tools")
        sequence_length = len(self.assistant_mask)
        return {
            "input_ids": torch.arange(sequence_length).unsqueeze(0),
            "assistant_masks": torch.tensor(self.assistant_mask).unsqueeze(0),
            "pixel_values": torch.ones(8, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "mm_token_type_ids": torch.zeros(1, sequence_length, dtype=torch.long),
        }


class NemotronH_Nano_Omni_Reasoning_V3Processor(_FakeQwenProcessor):
    model_input_names = ["input_ids", "pixel_values", "imgs_sizes", "num_frames"]

    def apply_chat_template(self, messages, **kwargs):
        if kwargs["tokenize"]:
            return "<image> prompt assistant"
        return "<image> prompt assistant"

    def __call__(self, **kwargs):
        return {
            "input_ids": torch.tensor([[10, 20, 21, 22]]),
            "pixel_values": torch.ones(2, 3, 4, 4),
            "imgs_sizes": torch.tensor([[4, 4], [4, 4]]),
            "num_frames": torch.ones(2, dtype=torch.long),
        }


def _sample(*, with_tools=False):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "media_index": 0},
                {"type": "text", "text": "Compare these."},
                {"type": "image", "media_index": 1},
            ],
        },
        {"role": "assistant", "content": "They match."},
    ]
    tools = None
    if with_tools:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "done",
                },
                {"role": "assistant", "content": "Final answer."},
            ]
        )
        tools = [{"type": "function", "function": {"name": "lookup"}}]
    return CanonicalSFTSample(
        __key__="sample-0",
        __restore_key__=("sample-0",),
        messages=messages,
        media=[
            MediaRef("image", torch.ones(3, 4, 4)),
            MediaRef("image", torch.zeros(3, 4, 4)),
        ],
        tools=tools,
    )


def test_qwen_adapter_preserves_media_order_tools_and_side_tensors():
    processor = _FakeQwenProcessor()
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=16,
        only_unmask_final=False,
    )

    encoded = adapter.encode(_sample(with_tools=True))

    user_content = processor.messages[0]["content"]
    assert [part["type"] for part in user_content] == ["image", "text", "image"]
    assert torch.equal(user_content[0]["image"], torch.ones(3, 4, 4))
    assert torch.equal(user_content[2]["image"], torch.zeros(3, 4, 4))
    assert processor.tools == [{"type": "function", "function": {"name": "lookup"}}]
    assert encoded.token_mask.tolist() == [0, 0, 1, 1]
    assert isinstance(encoded.model_inputs["pixel_values"], PackedTensor)
    assert isinstance(encoded.model_inputs["image_grid_thw"], PackedTensor)
    assert encoded.model_inputs["mm_token_type_ids"].shape == (4,)


def test_only_unmask_final_keeps_last_assistant_span():
    processor = _FakeQwenProcessor([0, 1, 1, 0, 1, 1])
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=16,
        only_unmask_final=True,
    )

    encoded = adapter.encode(_sample())

    assert encoded.token_mask.tolist() == [0, 0, 0, 0, 1, 1]


def test_nano_adapter_keeps_placeholder_side_tensors(monkeypatch):
    processor = NemotronH_Nano_Omni_Reasoning_V3Processor()
    monkeypatch.setattr(
        processor.tokenizer,
        "apply_chat_template",
        lambda messages, **kwargs: {
            "input_ids": torch.tensor([[10, 20, 21, 22]]),
            "assistant_masks": torch.tensor([[0, 0, 1, 1]]),
        },
        raising=False,
    )
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=16,
        only_unmask_final=False,
    )

    encoded = adapter.encode(_sample())

    assert encoded.token_mask.tolist() == [0, 0, 1, 1]
    assert isinstance(encoded.model_inputs["pixel_values"], PackedTensor)
    assert isinstance(encoded.model_inputs["imgs_sizes"], PackedTensor)
    assert isinstance(encoded.model_inputs["num_frames"], PackedTensor)


def test_task_encoder_batches_heterogeneous_rows():
    processor = _FakeQwenProcessor()
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=16,
        only_unmask_final=False,
    )
    encoder = EnergonSFTTaskEncoder(
        adapter=adapter,
        pad_token_id=0,
        sequence_length_pad_multiple=8,
    )
    assert encoder.decoder.config()["image_decode"] == "pilrgb"
    encoded = adapter.encode(_sample())
    text_only = deepcopy(encoded)
    text_only.model_inputs = {"mm_token_type_ids": torch.zeros(4, dtype=torch.long)}

    batch = encoder.batch([encoded, text_only])

    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_lengths"].tolist() == [4, 4]
    assert len(batch["pixel_values"]) == 2
    assert batch["pixel_values"].tensors[1] is None


class _FakeLoader:
    def __init__(self):
        self.restored = None

    def __iter__(self):
        return iter([{"value": 1}])

    def __len__(self):
        return 1

    def save_state_rank(self):
        return {"next": 1}

    def restore_state_rank(self, state):
        self.restored = state


def test_loader_state_is_fingerprinted_and_restored_before_iteration():
    raw_loader = _FakeLoader()
    loader = EnergonSFTDataLoader(raw_loader, fingerprint="expected")
    state = loader.state_dict()
    checkpoint = io.BytesIO()
    torch.save(state, checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)

    restored_raw_loader = _FakeLoader()
    restored = EnergonSFTDataLoader(restored_raw_loader, fingerprint="expected")
    restored.load_state_dict(state)

    assert restored_raw_loader.restored == {"next": 1}
    assert list(restored) == [{"value": 1}]
    with pytest.raises(RuntimeError, match="before iteration"):
        restored.load_state_dict(state)

    mismatched = EnergonSFTDataLoader(_FakeLoader(), fingerprint="changed")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        mismatched.load_state_dict(state)


def test_energon_config_defaults_and_validation():
    assert EnergonLoaderConfig().packing_buffer_size is None
    assert EnergonLoaderConfig().processor_adapter == "hf_multimodal"
    source = EnergonSourceConfig(
        path="/data/prepared", split="train", virtual_epoch_length=10
    )
    assert source.virtual_epoch_length == 10
    with pytest.raises(ValueError):
        EnergonLoaderConfig(packing_buffer_size=10)
