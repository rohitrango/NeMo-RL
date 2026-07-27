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

import importlib.util
import threading
from pathlib import Path

from nemo_rl.models.generation.vllm import patches


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message, *args):
        self.info_messages.append(message % args if args else message)

    def warning(self, message, *args):
        self.warning_messages.append(message % args if args else message)


def _stock_sampling_params_source() -> str:
    return """from __future__ import annotations

import copy
import json as json_mod
from dataclasses import field

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2

class SamplingParams:
    def update_from_tokenizer(self, tokenizer: TokenizerLike) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)

        invalid_token_ids = [
            token_id
            for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise VLLMValidationError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.",
                parameter="bad_words",
                value=self.bad_words,
            )
"""


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("patched_sampling_params", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CountingTokenizer:
    max_token_id = 10000

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def encode(self, *, text, add_special_tokens):
        assert add_special_tokens is False
        with self._lock:
            self.calls += 1
        return [len(text), sum(map(ord, text)) % self.max_token_id]


def _sampling_params(module, bad_words):
    value = module.SamplingParams()
    value.bad_words = bad_words
    return value


def test_bad_words_patch_cache_is_shared_bounded_and_thread_safe(tmp_path, monkeypatch):
    sampling_params = tmp_path / "sampling_params.py"
    sampling_params.write_text(_stock_sampling_params_source())
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(sampling_params))
    logger = _Logger()

    patches._patch_vllm_bad_words_tokenization_cache(logger)
    patched = sampling_params.read_text()

    assert "import threading" in patched
    assert "_BAD_WORDS_TOKEN_IDS_CACHE_MAX_ENTRIES = 1024" in patched
    assert "with _BAD_WORDS_TOKEN_IDS_CACHE_LOCK:" in patched
    assert "def _tokenize_bad_words" in patched
    assert "return bad_words_token_ids" in patched
    assert not logger.warning_messages

    module = _load_module(sampling_params)
    tokenizer = _CountingTokenizer()
    results = []
    errors = []
    start = threading.Barrier(8)

    def update_from_tokenizer():
        try:
            value = _sampling_params(module, ["<image>", "<img>"])
            start.wait()
            value.update_from_tokenizer(tokenizer)
            results.append(value._bad_words_token_ids)
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=update_from_tokenizer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 8
    assert all(result == results[0] for result in results)
    assert tokenizer.calls == 4
    assert len(module._BAD_WORDS_TOKEN_IDS_CACHE) == 1

    module._BAD_WORDS_TOKEN_IDS_CACHE.clear()
    for index in range(module._BAD_WORDS_TOKEN_IDS_CACHE_MAX_ENTRIES + 1):
        value = _sampling_params(module, [f"word-{index}"])
        value.update_from_tokenizer(tokenizer)
    assert (
        0
        < len(module._BAD_WORDS_TOKEN_IDS_CACHE)
        <= module._BAD_WORDS_TOKEN_IDS_CACHE_MAX_ENTRIES
    )
    assert (id(tokenizer), ("word-0",)) not in module._BAD_WORDS_TOKEN_IDS_CACHE
    assert (id(tokenizer), ("word-1024",)) in module._BAD_WORDS_TOKEN_IDS_CACHE

    patches._patch_vllm_bad_words_tokenization_cache(logger)
    assert sampling_params.read_text() == patched
    assert (
        logger.info_messages[-1] == "vLLM bad_words tokenization cache already applied."
    )


def test_bad_words_patch_leaves_unknown_vllm_source_untouched(tmp_path, monkeypatch):
    sampling_params = Path(tmp_path) / "sampling_params.py"
    sampling_params.write_text("# newer vLLM source\n")
    monkeypatch.setattr(patches, "_get_vllm_file", lambda _: str(sampling_params))
    logger = _Logger()

    patches._patch_vllm_bad_words_tokenization_cache(logger)

    assert sampling_params.read_text() == "# newer vLLM source\n"
    assert "expected vLLM 0.20.0 source shape" in logger.warning_messages[-1]
