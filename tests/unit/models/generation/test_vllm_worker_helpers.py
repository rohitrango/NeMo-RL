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

"""Tests for vLLM worker helper functions."""

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.vllm_worker import (
    _apply_nemotron_omni_layer_truncation,
)
from nemo_rl.models.generation.vllm.worker_utils import (
    find_tokenizer_required_architectures,
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
)


def test_nemotron_omni_layer_truncation_builds_nested_hf_override():
    llm_config = SimpleNamespace(
        layers_block_type=["mamba", "moe", "attention", "moe"],
        to_dict=lambda: {
            "layers_block_type": ["mamba", "moe", "attention", "moe"],
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["attention", "moe"],
        },
    )
    vllm_kwargs = {
        "nemo_truncate_num_layers": 2,
        "hf_overrides": {"max_position_embeddings": 1024},
    }

    _apply_nemotron_omni_layer_truncation(
        vllm_kwargs, SimpleNamespace(llm_config=llm_config)
    )

    assert "nemo_truncate_num_layers" not in vllm_kwargs
    target_llm_config = SimpleNamespace(
        layers_block_type=["mamba", "moe", "attention", "moe"],
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
    )
    target_config = SimpleNamespace(
        llm_config=target_llm_config,
        update=lambda values: vars(target_config).update(values),
    )
    result = vllm_kwargs["hf_overrides"](target_config)

    assert result.max_position_embeddings == 1024
    assert result.llm_config.layers_block_type == ["mamba", "moe"]
    assert result.llm_config.num_nextn_predict_layers == 0
    assert result.llm_config.mtp_layers_block_type == []


@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (None, []),
        ([], []),
        (["Gemma4ForCausalLM"], []),
        (
            ["Gemma4ForConditionalGeneration"],
            ["Gemma4ForConditionalGeneration"],
        ),
        (
            [
                "Gemma4ForCausalLM",
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
            [
                "Gemma4UnifiedForConditionalGeneration",
                "Mistral3ForConditionalGeneration",
            ],
        ),
    ],
)
def test_find_tokenizer_required_architectures(architectures, expected):
    assert find_tokenizer_required_architectures(architectures) == expected


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )
