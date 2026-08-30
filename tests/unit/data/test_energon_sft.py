import hashlib
import io
import json
from copy import deepcopy

import pytest
import torch

from nemo_rl.algorithms.sft import prepare_sft_batch
from nemo_rl.data.energon.config import (
    EnergonLoaderConfig,
    EnergonSourceConfig,
)
from nemo_rl.data.energon.multimodal.cookers.generic import cook_conversation
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    GenericSFTTaskEncoder,
    HFMultimodalSFTProcessorAdapter,
)
from nemo_rl.data.energon.multimodal.types import CanonicalSFTSample, MediaRef
from nemo_rl.data.energon.sft_dataloader import (
    EnergonSFTDataLoader,
    _loader_config,
    _v1_fingerprint,
)
from nemo_rl.data.llm_message_utils import message_log_to_flat_messages
from nemo_rl.data.multimodal_utils import PackedTensor


class _FakeTokenizer:
    pad_token_id = 0
    bos_token = None
    eos_token = "<eos>"
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
    pad_token_id = 0
    bos_token = None
    eos_token = "<eos>"

    def __init__(self):
        self.messages = None
        self.tools = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = deepcopy(messages)
        self.tools = kwargs.get("tools")
        assert kwargs["tokenize"] is False

        def render_content(content):
            if isinstance(content, str):
                return content
            return "".join(
                part.get("text", "") if part["type"] == "text" else f"<{part['type']}>"
                for part in content
            )

        return "".join(
            f"<{message['role']}>{render_content(message.get('content', ''))}"
            f"</{message['role']}>"
            for message in messages
        )

    def __call__(self, *, text, images=None, **kwargs):
        text = text[0] if isinstance(text, list) else text
        input_ids = torch.tensor(
            [[ord(character) % 251 + 1 for character in text]], dtype=torch.long
        )
        processed = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.zeros_like(input_ids),
        }
        if images:
            processed["pixel_values"] = torch.ones(len(images), 4)
            processed["image_grid_thw"] = torch.tensor(
                [[1, 2, 2]] * len(images), dtype=torch.long
            )
        return processed


class NemotronH_Nano_Omni_Reasoning_V3Processor(_FakeQwenProcessor):
    model_input_names = ["input_ids", "pixel_values", "imgs_sizes", "num_frames"]

    def __call__(self, *, text, images=None, **kwargs):
        processed = {"input_ids": torch.tensor([[10, 20, 21, 22]])}
        if images:
            processed["pixel_values"] = torch.ones(len(images), 3, 4, 4)
        return processed


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


def _adapter(processor, *, max_sequence_length=1024):
    return HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=max_sequence_length,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=False,
    )


def _encoder(adapter, *, include_source_ids=False):
    return GenericSFTTaskEncoder(
        adapter=adapter,
        cooker_functions=[cook_conversation],
        include_source_ids=include_source_ids,
    )


def test_qwen_adapter_returns_tokenized_message_log_with_model_inputs():
    processor = _FakeQwenProcessor()
    encoded = _adapter(processor).encode(_sample(with_tools=True))

    user_content = processor.messages[0]["content"]
    assert [part["type"] for part in user_content] == ["image", "text", "image"]
    assert torch.equal(user_content[0]["image"], torch.ones(3, 4, 4))
    assert torch.equal(user_content[2]["image"], torch.zeros(3, 4, 4))
    assert processor.tools == [{"type": "function", "function": {"name": "lookup"}}]

    flat = message_log_to_flat_messages(encoded.message_log)
    assert [message["role"] for message in encoded.message_log] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]
    assert isinstance(flat["pixel_values"], PackedTensor)
    assert isinstance(flat["image_grid_thw"], PackedTensor)
    assert flat["mm_token_type_ids"].shape == flat["token_ids"].shape


def test_prepare_sft_batch_builds_mask_from_energon_message_log():
    processor = _FakeQwenProcessor()
    adapter = _adapter(processor)
    encoder = _encoder(adapter)
    encoded = adapter.encode(_sample())

    batch = encoder.batch([encoded])
    prepared = prepare_sft_batch(
        batch,
        tokenizer=processor,
        only_unmask_final=False,
        make_sequence_length_divisible_by=8,
    )

    expected_ids = torch.cat([message["token_ids"] for message in encoded.message_log])
    expected_mask = torch.cat(
        [
            torch.full_like(message["token_ids"], int(message["role"] == "assistant"))
            for message in encoded.message_log
        ]
    )
    assert torch.equal(prepared["input_ids"][0, : encoded.length], expected_ids)
    assert torch.equal(prepared["token_mask"][0, : encoded.length], expected_mask)
    assert prepared["input_ids"].shape[1] % 8 == 0


def test_nano_adapter_keeps_placeholder_side_tensors():
    encoded = _adapter(NemotronH_Nano_Omni_Reasoning_V3Processor()).encode(_sample())

    flat = message_log_to_flat_messages(encoded.message_log)
    assert isinstance(flat["pixel_values"], PackedTensor)
    assert isinstance(flat["imgs_sizes"], PackedTensor)
    assert isinstance(flat["num_frames"], PackedTensor)


def test_task_encoder_batches_heterogeneous_message_logs():
    processor = _FakeQwenProcessor()
    adapter = _adapter(processor)
    encoder = _encoder(adapter)
    assert encoder.decoder.config()["image_decode"] == "pilrgb"

    encoded = adapter.encode(_sample())
    text_only = deepcopy(encoded)
    for message in text_only.message_log:
        message["content"] = "text only"
        for key in ("pixel_values", "image_grid_thw"):
            message.pop(key, None)

    batch = encoder.batch([encoded, text_only])
    prepared = prepare_sft_batch(
        batch,
        tokenizer=processor,
        only_unmask_final=False,
        make_sequence_length_divisible_by=8,
    )

    assert list(batch) == ["message_log", "loss_multiplier"]
    assert prepared["input_ids"].shape[0] == 2
    assert len(prepared["pixel_values"]) == 2
    assert prepared["pixel_values"].tensors[1] is None


def test_task_encoder_runs_split_encode_and_batch_lifecycle_methods():
    adapter = _adapter(_FakeQwenProcessor())
    encoder = _encoder(adapter, include_source_ids=True)

    preencoded = encoder.preencode_sample(_sample())
    postencoded = encoder.postencode_sample(preencoded)
    batch = encoder.batch([postencoded])

    assert encoder.encode_batch(batch) is batch
    assert batch["source_ids"] == ["sample-0"]
    with pytest.raises(RuntimeError, match="No Energon packing"):
        encoder.select_samples_to_pack([preencoded])


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


def test_energon_config_disables_sequence_packing():
    config = EnergonLoaderConfig(model_family="qwen")
    assert config.model_family == "qwen"
    assert config.packing_buffer_size is None
    assert config.max_samples_per_sequence is None
    assert config.processor_adapter == "hf_multimodal"
    assert config.topology_mapper == "default"
    assert config.task_encoder.name == "generic_sft"
    assert [cooker.name for cooker in config.cookers] == ["generic_conversation"]

    source = EnergonSourceConfig(
        path="/data/prepared", split="train", virtual_epoch_length=10
    )
    assert source.virtual_epoch_length == 10
    with pytest.raises(ValueError):
        EnergonLoaderConfig(model_family="qwen", packing_buffer_size=10)
    with pytest.raises(ValueError):
        EnergonLoaderConfig(model_family="qwen", max_samples_per_sequence=2)
    with pytest.raises(ValueError):
        EnergonLoaderConfig.model_validate({})
    with pytest.raises(ValueError):
        EnergonLoaderConfig.model_validate({"model_family": "unsupported"})


def test_v1_fingerprint_uses_the_former_loader_fields_only():
    source = EnergonSourceConfig(
        path="/data/prepared", split="train", virtual_epoch_length=10
    )
    config = EnergonLoaderConfig(model_family="qwen")
    former_loader_config = {
        "num_workers": 8,
        "shuffle_buffer_size": 1000,
        "max_samples_per_sequence": None,
        "packing_buffer_size": None,
        "batch_grouping": "auto",
        "processor_adapter": "hf_multimodal",
        "seed_offset": 0,
        "prefetch_factor": 2,
        "checkpoint_every_sec": 60.0,
        "watchdog_timeout_seconds": 60.0,
    }
    payload = {
        "source": source.model_dump(mode="json"),
        "loader": former_loader_config,
        "adapter": "adapter-fingerprint",
        "split_role": "train",
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    assert (
        _v1_fingerprint(
            source=source,
            loader_config=config,
            adapter_fingerprint="adapter-fingerprint",
            split_role="train",
        )
        == expected
    )


def test_v1_fingerprint_identifies_stage3_component_selection():
    source = EnergonSourceConfig(path="/data/prepared", split="train")
    generic = EnergonLoaderConfig(model_family="qwen")
    stage3 = EnergonLoaderConfig.model_validate(
        {
            "model_family": "qwen",
            "cookers": [
                {
                    "name": "generic_conversation",
                    "has_subflavors": {"cook": "conversation"},
                }
            ],
        }
    )

    fingerprints = {
        _v1_fingerprint(
            source=source,
            loader_config=config,
            adapter_fingerprint="same-processor",
            split_role="train",
        )
        for config in (generic, stage3)
    }

    assert len(fingerprints) == 2


def test_config_parses_registry_keys_and_validates_packing_options():
    config = EnergonLoaderConfig.model_validate(
        {
            "model_family": "qwen",
            "task_encoder": "generic_sft",
            "cookers": ["generic_conversation"],
        }
    )
    assert config.task_encoder.name == "generic_sft"
    assert config.cookers[0].name == "generic_conversation"


def test_config_rejects_options_for_the_generic_task_encoder():
    with pytest.raises(ValueError, match="has no configurable options"):
        _loader_config(
            {
                "model_family": "qwen",
                "task_encoder": {
                    "name": "generic_sft",
                    "options": {"patch_dim": 14},
                },
            }
        )
