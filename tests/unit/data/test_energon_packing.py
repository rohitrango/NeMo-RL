from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonPackingOptions
from nemo_rl.data.energon.multimodal.packing.sft import (
    build_packing_hooks,
    select_samples_to_pack,
)
from nemo_rl.data.energon.multimodal.packing.prepare import (
    prepare_energon_packed_batch,
)
from nemo_rl.data.energon.multimodal.registry import PACKING_REGISTRY
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    GenericSFTTaskEncoder,
)
from nemo_rl.data.energon.multimodal.types import EncodedSFTSample
from nemo_rl.data.energon.sft_dataloader import (
    _task_encoder,
    build_energon_sft_loader,
    build_energon_sft_dataloaders,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data.packing import FirstFitDecreasingPacker


class _Tokenizer:
    pad_token_id = 0


class _Adapter:
    def encode(self, sample):
        raise AssertionError("These tests use pre-encoded samples.")


def _sample(
    key: str,
    length: int,
    *,
    group: str = "text",
    packing_cost: int | None = None,
    with_media: bool = False,
) -> EncodedSFTSample:
    user_length = max(1, length - 2)
    assistant_length = length - user_length
    user = {
        "role": "user",
        "token_ids": torch.arange(1, user_length + 1),
    }
    if with_media:
        user["pixel_values"] = PackedTensor(
            torch.full((1, 2), int(key[-1]) + 1, dtype=torch.float32),
            dim_to_pack=0,
        )
    return EncodedSFTSample(
        __key__=key,
        __restore_key__=(key,),
        message_log=[
            user,
            {
                "role": "assistant",
                "token_ids": torch.arange(
                    user_length + 1, user_length + assistant_length + 1
                ),
            },
        ],
        length=length,
        packing_cost=length if packing_cost is None else packing_cost,
        loss_multiplier=1.0,
        group_key=(group,),
        sample_key=key,
    )


def _options(*, capacity: int = 8, alignment: int = 1) -> EnergonPackingOptions:
    return EnergonPackingOptions(
        max_sequence_length=capacity,
        sequence_length_pad_multiple=alignment,
    )


def _hooks(*, capacity: int = 8, alignment: int = 1):
    return build_packing_hooks(
        _options(capacity=capacity, alignment=alignment),
        algorithm="first_fit_decreasing",
        version="1",
        packer_type=FirstFitDecreasingPacker,
    )


def test_first_fit_is_deterministic_complete_and_capacity_bounded() -> None:
    samples = [
        _sample("s0", 6),
        _sample("s1", 5),
        _sample("s2", 4),
        _sample("s3", 3),
    ]

    first = select_samples_to_pack(
        samples,
        packer=FirstFitDecreasingPacker(8),
        sequence_length_pad_multiple=1,
    )
    second = select_samples_to_pack(
        samples,
        packer=FirstFitDecreasingPacker(8),
        sequence_length_pad_multiple=1,
    )

    assert [[sample.sample_key for sample in pack] for pack in first] == [
        ["s0"],
        ["s1", "s3"],
        ["s2"],
    ]
    assert [[sample.sample_key for sample in pack] for pack in second] == [
        ["s0"],
        ["s1", "s3"],
        ["s2"],
    ]
    assert sorted(sample.sample_key for pack in first for sample in pack) == [
        "s0",
        "s1",
        "s2",
        "s3",
    ]
    assert all(sum(sample.length for sample in pack) <= 8 for pack in first)


def test_first_fit_respects_alignment_groups_and_oversized_sources() -> None:
    samples = [_sample("s0", 3, group="a"), _sample("s1", 3, group="b")]
    packs = select_samples_to_pack(
        samples,
        packer=FirstFitDecreasingPacker(8),
        sequence_length_pad_multiple=4,
    )

    assert len(packs) == 2
    assert [pack[0].group_key for pack in packs] == [("a",), ("b",)]
    with pytest.raises(ValueError, match="exceeds bin capacity"):
        select_samples_to_pack(
            [_sample("large", 9)],
            packer=FirstFitDecreasingPacker(8),
            sequence_length_pad_multiple=1,
        )


def test_energon_selection_matches_shared_packer_indexes() -> None:
    samples = [
        _sample("s0", 3, packing_cost=5),
        _sample("s1", 4, packing_cost=4),
        _sample("s2", 2, packing_cost=3),
    ]
    packer = FirstFitDecreasingPacker(12)

    direct_indexes = packer.pack([8, 4, 4])
    energon_packs = select_samples_to_pack(
        samples,
        packer=packer,
        sequence_length_pad_multiple=4,
    )

    assert [[sample.sample_key for sample in pack] for pack in energon_packs] == [
        [samples[index].sample_key for index in indexes] for indexes in direct_indexes
    ]


def test_registered_hooks_build_packed_sample_and_reject_mixed_groups() -> None:
    packer_type = PACKING_REGISTRY.resolve("first_fit_decreasing")
    hooks = build_packing_hooks(
        _options(capacity=16, alignment=4),
        algorithm="first_fit_decreasing",
        version="1",
        packer_type=packer_type,
    )
    source_a = _sample("s0", 5)
    source_b = _sample("s1", 3)

    packed = hooks.pack_selected_samples([source_a, source_b])

    assert hooks.sample_schema == GenericSFTTaskEncoder.sample_schema
    assert packed.source_ids == ["s0", "s1"]
    assert packed.source_lengths == [5, 3]
    assert packed.source_padded_lengths == [8, 4]
    with pytest.raises(ValueError, match="compatibility groups"):
        hooks.pack_selected_samples([source_a, _sample("other", 2, group="image")])


def test_physical_pack_preserves_repeated_source_ids() -> None:
    hooks = _hooks(capacity=16)

    packed = hooks.pack_selected_samples([_sample("same", 4), _sample("same", 4)])

    assert packed.source_ids == ["same", "same"]


def test_task_encoder_packed_lifecycle_and_preparation_preserve_boundaries() -> None:
    hooks = _hooks(capacity=16, alignment=4)
    encoder = GenericSFTTaskEncoder(
        adapter=_Adapter(),
        cooker_functions=[],
        packing_hooks=hooks,
        include_source_ids=True,
    )
    sources = [
        _sample("s0", 5, with_media=True),
        _sample("s1", 3, with_media=True),
    ]
    sources[0].message_log[0]["mm_token_type_ids"] = torch.tensor([1, 2, 3])
    selected = encoder.select_samples_to_pack(sources)
    physical = [
        encoder.pack_selected_samples(
            [encoder.postencode_sample(item) for item in pack]
        )
        for pack in selected
    ]
    encoded_batch = encoder.encode_batch(encoder.batch(physical))

    prepared = prepare_energon_packed_batch(
        encoded_batch,
        tokenizer=_Tokenizer(),
        only_unmask_final=False,
    )

    assert prepared["packed_schema_version"].tolist() == [1]
    assert prepared["source_ids"] == [["s0", "s1"]]
    assert prepared["source_lengths"] == [[5, 3]]
    assert prepared["input_lengths"].tolist() == [16]
    assert torch.equal(prepared["cu_seqlens"][0], torch.tensor([0, 5, 8]))
    assert torch.equal(prepared["cu_seqlens_padded"][0], torch.tensor([0, 8, 16]))
    assert prepared["input_ids"].shape == prepared["token_mask"].shape == (1, 16)
    assert prepared["mm_token_type_ids"][0].tolist() == [
        1,
        2,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert prepared["token_mask"][0].tolist() == [
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    assert len(prepared["pixel_values"]) == 1
    assert torch.equal(
        prepared["pixel_values"].as_tensor(),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )


def test_expanded_packing_cost_sets_aligned_physical_boundaries() -> None:
    hooks = _hooks(capacity=12, alignment=4)
    sources = [
        _sample("s0", 3, packing_cost=5),
        _sample("s1", 3, packing_cost=4),
    ]
    selected = hooks.select_samples_to_pack(sources)
    packed = hooks.pack_selected_samples(selected[0])
    encoder = GenericSFTTaskEncoder(
        adapter=_Adapter(),
        cooker_functions=[],
        packing_hooks=hooks,
        include_source_ids=True,
    )

    prepared = prepare_energon_packed_batch(
        encoder.batch([packed]),
        tokenizer=_Tokenizer(),
        only_unmask_final=False,
    )

    assert packed.source_lengths == [3, 3]
    assert packed.source_padded_lengths == [8, 4]
    assert prepared["source_lengths"] == [[3, 3]]
    assert prepared["input_lengths"].tolist() == [12]
    assert prepared["cu_seqlens"][0].tolist() == [0, 3, 6]
    assert prepared["cu_seqlens_padded"][0].tolist() == [0, 8, 12]


def test_only_unmask_final_is_applied_per_source_before_packing() -> None:
    source_a = _sample("s0", 4)
    source_a.message_log.append(
        {"role": "assistant", "token_ids": torch.tensor([9, 10])}
    )
    source_a.length += 2
    source_a.packing_cost += 2
    source_b = _sample("s1", 4)
    hooks = _hooks(capacity=16, alignment=1)
    packed = hooks.pack_selected_samples([source_a, source_b])
    encoder = GenericSFTTaskEncoder(
        adapter=_Adapter(),
        cooker_functions=[],
        packing_hooks=hooks,
        include_source_ids=True,
    )

    prepared = prepare_energon_packed_batch(
        encoder.batch([packed]),
        tokenizer=_Tokenizer(),
        only_unmask_final=True,
    )

    assert prepared["token_mask"].sum().item() == 4
    assert prepared["token_mask"][0, 4:6].tolist() == [1, 1]
    assert prepared["token_mask"][0, 8:10].tolist() == [1, 1]


def test_loader_resolves_hooks_and_v1_rejects_energon_packing() -> None:
    loader_config = EnergonLoaderConfig.model_validate(
        {
            "model_family": "qwen",
            "cookers": [
                {
                    "name": "generic_conversation",
                    "has_subflavors": {"source_schema": "openai"},
                }
            ],
            "task_encoder": {
                "packing": {
                    "name": "first_fit_decreasing",
                    "buffer_size": 32,
                    "options": {
                        "max_sequence_length": 128,
                        "sequence_length_pad_multiple": 8,
                    },
                }
            },
        }
    )
    encoder = _task_encoder(
        loader_config=loader_config,
        adapter=_Adapter(),
        include_source_ids=True,
    )
    assert encoder.select_samples_to_pack([_sample("s0", 4)])[0][0].sample_key == "s0"
    assert encoder.cookers[0].has_subflavors == {"source_schema": "openai"}

    data_config = {
        "energon": loader_config.model_dump(mode="python"),
        "shuffle": True,
        "train": {
            "path": "/dataset",
            "split": "train",
            "virtual_epoch_length": 8,
        },
        "validation": None,
    }
    with pytest.raises(ValueError, match="requires the SFTv2 loader"):
        build_energon_sft_dataloaders(
            data_config=data_config,
            processor=MagicMock(),
            train_batch_size=1,
            val_batch_size=1,
            max_sequence_length=128,
        )


@patch("nemo_rl.data.energon.sft_dataloader.get_savable_loader")
@patch("nemo_rl.data.energon.sft_dataloader.get_train_dataset")
def test_v2_loader_passes_registered_packing_buffer_to_energon(
    get_train_dataset: MagicMock, get_savable_loader: MagicMock
) -> None:
    get_train_dataset.return_value = MagicMock()
    get_savable_loader.return_value = MagicMock()
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    data_config = {
        "shuffle": True,
        "energon": {
            "model_family": "qwen",
            "task_encoder": {
                "packing": {
                    "name": "first_fit_decreasing",
                    "buffer_size": 32,
                    "options": {
                        "max_sequence_length": 128,
                        "sequence_length_pad_multiple": 8,
                    },
                }
            },
        },
    }

    build_energon_sft_loader(
        data_config=data_config,
        source={
            "path": "/dataset",
            "split": "train",
            "virtual_epoch_length": 8,
        },
        processor=processor,
        batch_size=2,
        max_sequence_length=128,
        split_role="train",
        logical_rank=0,
        logical_world_size=1,
        placement_fingerprint="placement",
    )

    assert get_train_dataset.call_args.kwargs["packing_buffer_size"] == 32
    task_encoder = get_train_dataset.call_args.kwargs["task_encoder"]
    assert (
        task_encoder.select_samples_to_pack([_sample("s0", 4)])[0][0].sample_key == "s0"
    )
    assert get_savable_loader.call_args.kwargs["cache_pool"] is None

