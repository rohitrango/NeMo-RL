import sys
from types import ModuleType

import pytest

from nemo_rl.data.energon.multimodal.registry import (
    COOKER_REGISTRY,
    PACKING_REGISTRY,
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
    assert PACKING_REGISTRY.identity("first_fit_decreasing") == {
        "key": "first_fit_decreasing",
        "version": "1",
    }
    assert selected_registry_identity(
        task_encoder="generic_sft",
        cookers=["generic_conversation"],
        packing=None,
    ) == {
        "task_encoder": {"key": "generic_sft", "version": "1"},
        "cookers": [{"key": "generic_conversation", "version": "1"}],
        "packing": None,
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
    registry = LazyRegistry("packing")
    registry.register(
        "invalid",
        import_path=f"{module.__name__}:invalid",
        version="1",
    )

    with pytest.raises(TypeError, match="SequencePacker subclass"):
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
