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

import sys
from types import ModuleType

import pytest

from nemo_rl.data.energon.multimodal.registry import (
    COOKER_REGISTRY,
    TASK_ENCODER_REGISTRY,
    LazyRegistry,
    selected_registry_identity,
)


def test_builtin_registries_resolve_lazily_with_stable_versions():
    assert COOKER_REGISTRY.identity("generic_conversation") == {
        "key": "generic_conversation",
        "version": "1",
    }
    assert TASK_ENCODER_REGISTRY.identity("generic_sft") == {
        "key": "generic_sft",
        "version": "1",
    }
    assert selected_registry_identity(
        task_encoder="generic_sft",
        cookers=["generic_conversation"],
    ) == {
        "task_encoder": {"key": "generic_sft", "version": "1"},
        "cookers": [{"key": "generic_conversation", "version": "1"}],
    }


def test_registry_rejects_duplicate_unknown_and_malformed_keys():
    registry = LazyRegistry("cooker")
    registry.register("one", import_path="module:function", version="1")

    with pytest.raises(ValueError, match="Duplicate cooker"):
        registry.register("one", import_path="other:function", version="2")
    with pytest.raises(ValueError, match="Unknown cooker"):
        registry.resolve("missing")
    with pytest.raises(ValueError, match="expected module:name"):
        registry.register("bad", import_path="module.function", version="1")


def test_registry_rejects_invalid_resolved_type(monkeypatch):
    module = ModuleType("_test_energon_registry_module")
    module.invalid = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LazyRegistry("task_encoder")
    registry.register(
        "invalid",
        import_path=f"{module.__name__}:invalid",
        version="1",
    )

    with pytest.raises(TypeError, match="BaseSFTTaskEncoder subclass"):
        registry.resolve("invalid")


def test_registry_does_not_import_component_during_registration(monkeypatch):
    imported: list[str] = []
    original = __import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", record_import)
    registry = LazyRegistry("cooker")
    registry.register("json_loads", import_path="json:loads", version="1")

    assert "json" not in imported
    assert registry.resolve("json_loads") is not None
