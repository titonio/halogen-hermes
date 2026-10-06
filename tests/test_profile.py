"""Provider profile tests: env parsing, Hermes hook wiring, registration.

``providers`` / ``providers.base`` come from the stub injected by
tests/conftest.py (or the real Hermes package, if present).
"""

import importlib.util
import pathlib
import sys

import pytest

import providers
from errors import classify_halogen_error
from profile import HalogenProfile, build_profile

MODEL = "halogen-qwen3.8-flash-next"
ROOT = pathlib.Path(__file__).resolve().parents[1]

needs_stub_registry = pytest.mark.skipif(
    not hasattr(providers, "REGISTRY"),
    reason="real Hermes providers package in use; stub registry absent",
)


def test_default_env_lands_on_defaults():
    p = build_profile({})
    assert p.name == "halogen"
    assert p.base_url == "http://127.0.0.1:8731/v1"
    assert p.fallback_models == (MODEL,)
    assert p.default_aux_model == MODEL
    assert p.default_max_tokens == 8192
    assert p.model_capabilities[MODEL] == {
        "supports_tools": True,
        "supports_reasoning": True,
        "supports_vision": False,
        "context_window": 262144,
    }


def test_declared_identity_fields():
    p = build_profile({})
    assert isinstance(p, HalogenProfile)
    assert p.aliases == ("halo",)
    assert p.env_vars == ("HALOGEN_API_KEY", "HALOGEN_BASE_URL")
    assert p.auth_type == "api_key"
    assert p.supports_vision is False
    assert p.display_name
    assert p.description


def test_env_overrides_land():
    p = build_profile({
        "HALOGEN_BASE_URL": "http://halogen.example:8731/v1",
        "HALOGEN_CTX": "1048576",
        "HALOGEN_VISION": "1",
        "HALOGEN_MAX_TOKENS": "65536",
    })
    assert p.base_url == "http://halogen.example:8731/v1"
    assert p.default_max_tokens == 65536
    assert p.supports_vision is True
    assert p.model_capabilities[MODEL]["supports_vision"] is True
    assert p.model_capabilities[MODEL]["context_window"] == 1048576
    assert p.get_model_context_length(MODEL) == 1048576


def test_malformed_env_falls_back_to_defaults():
    # Review Focus 3: a bad environment must never crash Hermes startup.
    p = build_profile({"HALOGEN_CTX": "abc", "HALOGEN_REASONING_EFFORT": "bogus"})
    assert p.get_model_context_length("anything") == 262144
    assert p.model_capabilities[MODEL]["context_window"] == 262144
    assert p.default_reasoning_config() == {"enabled": True, "effort": "medium"}


def test_supported_reasoning_efforts():
    assert build_profile({}).supported_reasoning_efforts(None) == (
        "none", "minimal", "low", "medium", "high", "xhigh",
    )


def test_default_reasoning_config_medium():
    assert build_profile({}).default_reasoning_config() == {"enabled": True, "effort": "medium"}


def test_default_reasoning_config_off():
    p = build_profile({"HALOGEN_REASONING_EFFORT": "off"})
    assert p.default_reasoning_config() == {"enabled": False}


def test_build_api_kwargs_extras_goes_to_extra_body():
    # Review Focus: wire fields ride in extra_body (first tuple element);
    # nothing may reach the top-level request kwargs.
    extra_body, top_level = build_profile({}).build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}
    )
    assert extra_body == {"enable_thinking": True, "reasoning_effort": "high"}
    assert top_level == {}


def test_build_api_kwargs_extras_absorbs_hermes_context_kwargs():
    # Real Hermes calls build_api_kwargs_extras(*, reasoning_config=..., model=..., ...)
    # — the override must accept the extra context kwargs.
    extra_body, top_level = build_profile({}).build_api_kwargs_extras(
        reasoning_config=None, model=MODEL, base_url="http://x/v1"
    )
    assert extra_body == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert top_level == {}


def test_preserve_thinking_env_flows_to_wire_fields():
    p = build_profile({"HALOGEN_PRESERVE_THINKING": "1"})
    extra_body, _ = p.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"})
    assert extra_body == {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "preserve_thinking": True,
    }


def test_build_client_kwargs_extras_keyless_placeholder():
    # Review Focus 5: client construction without any credential still works.
    assert build_profile({}).build_client_kwargs_extras() == {"api_key": "no-key-required"}


def test_build_client_kwargs_extras_uses_configured_key():
    p = build_profile({"HALOGEN_API_KEY": "secret-token"})
    assert p.build_client_kwargs_extras() == {"api_key": "secret-token"}


def test_get_max_tokens_clamped_to_quarter_ctx():
    assert build_profile({"HALOGEN_MAX_TOKENS": "100000"}).get_max_tokens(None) == 65536


def test_get_max_tokens_default_below_clamp_passes_through():
    assert build_profile({}).get_max_tokens(None) == 8192


def test_get_model_context_length_declared_ctx():
    assert build_profile({}).get_model_context_length("anything") == 262144


def test_classify_api_error_field_is_halogen_classifier():
    assert build_profile({}).classify_api_error is classify_halogen_error


def test_module_level_halogen_instance():
    import profile as profile_module
    assert isinstance(profile_module.halogen, HalogenProfile)
    assert profile_module.halogen.name == "halogen"


@needs_stub_registry
def test_package_import_registers_provider():
    # Mirror Hermes' user-plugin loader: __init__.py as a package with the
    # plugin dir as its submodule search path.
    module_name = "halogen_plugin_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        assert providers.REGISTRY["halogen"] is module.halogen
        assert providers.REGISTRY["halo"] is module.halogen
    finally:
        for name in [n for n in sys.modules if n.startswith(module_name)]:
            sys.modules.pop(name, None)
