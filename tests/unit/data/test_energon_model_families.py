import sys
from types import ModuleType

import pytest

from nemo_rl.data.energon.multimodal.model_families import (
    ALL_MODEL_FAMILIES,
    get_supported_model_families,
    supports_model_families,
    supports_model_family,
)
from nemo_rl.data.energon.multimodal.registry import (
    COOKER_REGISTRY,
    TASK_ENCODER_REGISTRY,
    LazyRegistry,
)
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (
    GenericSFTTaskEncoder,
)
from nemo_rl.data.energon.sft_dataloader import _loader_config


def test_decorator_stores_immutable_model_family_metadata() -> None:
    @supports_model_families("qwen", "nemotron")
    def component() -> None:
        pass

    supported = get_supported_model_families(component)

    assert supported == frozenset({"qwen", "nemotron"})
    assert supports_model_family(component, "qwen")
    assert supports_model_family(component, "nemotron")


def test_all_model_families_marker_supports_each_known_family() -> None:
    @supports_model_families(ALL_MODEL_FAMILIES)
    class Component:
        pass

    assert supports_model_family(Component, "qwen")
    assert supports_model_family(Component, "nemotron")


def test_builtin_generic_components_support_all_model_families() -> None:
    cooker = COOKER_REGISTRY.resolve_for_model_family(
        "generic_conversation",
        model_family="qwen",
    )

    assert get_supported_model_families(cooker) == frozenset({ALL_MODEL_FAMILIES})
    assert get_supported_model_families(GenericSFTTaskEncoder) == frozenset(
        {ALL_MODEL_FAMILIES}
    )


@pytest.mark.parametrize(
    ("key", "family"),
    [
        ("nemotron_multimodal", "nemotron"),
    ],
)
def test_model_specific_task_encoders_support_their_selected_family(
    key: str,
    family: str,
) -> None:
    resolved = TASK_ENCODER_REGISTRY.resolve_for_model_family(
        key,
        model_family=family,  # type: ignore[arg-type]
    )

    assert get_supported_model_families(resolved) == frozenset({family})


def test_model_family_metadata_rejects_missing_and_invalid_declarations() -> None:
    with pytest.raises(TypeError, match="no supported model-family declaration"):
        get_supported_model_families(lambda: None)
    with pytest.raises(ValueError, match="At least one"):
        supports_model_families()
    with pytest.raises(ValueError, match="Unknown model families"):
        supports_model_families("unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be combined"):
        supports_model_families(ALL_MODEL_FAMILIES, "qwen")


def test_registry_rejects_unsupported_and_undeclared_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("_test_energon_model_family_module")

    @supports_model_families("qwen")
    def qwen_only() -> None:
        pass

    module.qwen_only = qwen_only
    module.undeclared = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LazyRegistry("cooker")
    registry.register(
        "qwen_only",
        import_path=f"{module.__name__}:qwen_only",
        version="1",
    )
    registry.register(
        "undeclared",
        import_path=f"{module.__name__}:undeclared",
        version="1",
    )

    assert (
        registry.resolve_for_model_family("qwen_only", model_family="qwen") is qwen_only
    )
    with pytest.raises(
        ValueError,
        match="Cooker registry key 'qwen_only'.*model family 'nemotron'.*qwen",
    ):
        registry.resolve_for_model_family("qwen_only", model_family="nemotron")
    with pytest.raises(TypeError, match="registry key 'undeclared'.*must declare"):
        registry.resolve_for_model_family("undeclared", model_family="qwen")


def test_loader_setup_rejects_an_unsupported_cooker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("_test_energon_loader_model_family_module")

    @supports_model_families("qwen")
    def qwen_only() -> None:
        pass

    module.qwen_only = qwen_only
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LazyRegistry("cooker")
    registry.register(
        "qwen_only",
        import_path=f"{module.__name__}:qwen_only",
        version="1",
    )
    monkeypatch.setattr(
        COOKER_REGISTRY,
        "resolve_for_model_family",
        registry.resolve_for_model_family,
    )

    with pytest.raises(
        ValueError,
        match="Cooker registry key 'qwen_only'.*model family 'nemotron'.*qwen",
    ):
        _loader_config(
            {
                "model_family": "nemotron",
                "cookers": ["qwen_only"],
            }
        )


def test_packing_registry_has_no_model_family_validation() -> None:
    registry = LazyRegistry("packing")
    registry.register("callable", import_path="json:loads", version="1")

    with pytest.raises(TypeError, match="Packing registry entries"):
        registry.resolve_for_model_family("callable", model_family="qwen")


def test_undecorated_subclass_does_not_inherit_model_family_metadata() -> None:
    @supports_model_families("qwen")
    class QwenComponent:
        pass

    class UndeclaredSubclass(QwenComponent):
        pass

    with pytest.raises(TypeError, match="no supported model-family declaration"):
        get_supported_model_families(UndeclaredSubclass)
