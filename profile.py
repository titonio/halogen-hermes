"""Halogen provider profile for Hermes.

halogen-flash-server is an OpenAI-compatible Qwen inference server with no
auth; this profile routes the ``halogen`` provider name at it: thinking
control travels in ``extra_body`` wire fields, output budgets are clamped to
a quarter of the declared context window, and halogen-specific errors map
onto Hermes failover reasons. All decision logic lives in ``wire.py`` and
``errors.py``; this module only adapts them to Hermes' ProviderProfile
contract and reads configuration.

Stdlib only, plus ``providers.base`` which Hermes provides at runtime (the
test conftest injects a stub when Hermes is absent).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from providers.base import ProviderProfile

try:  # Hermes loads this file as a plugin-package submodule
    from .errors import classify_halogen_error
    from .wire import EFFORT_LEVELS, clamp_output_budget, resolve_thinking
except ImportError:  # standalone import (tests, scripts): repo root on sys.path
    from errors import classify_halogen_error
    from wire import EFFORT_LEVELS, clamp_output_budget, resolve_thinking

MODEL_ID = "halogen-qwen3.8-flash-next"
DEFAULT_BASE_URL = "http://127.0.0.1:8731/v1"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_CTX = 262144
KEYLESS_API_KEY = "no-key-required"

# Hermes' reasoning-effort vocabulary this route accepts (tri-state contract:
# a non-empty tuple makes the transport clamp requests onto exactly these).
REASONING_EFFORTS: tuple[str, ...] = ("none",) + EFFORT_LEVELS

_EFFORT_CHOICES: tuple[str, ...] = EFFORT_LEVELS + ("off",)


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = (env.get(name) or "").strip()
    return raw or default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _env_effort(env: Mapping[str, str], name: str, default: str) -> str:
    raw = (env.get(name) or "").strip().lower()
    return raw if raw in _EFFORT_CHOICES else default


def _env_bool(env: Mapping[str, str], name: str) -> bool:
    return (env.get(name) or "").strip() == "1"


class HalogenProfile(ProviderProfile):
    """halogen-flash-server route: thinking wire fields, budget clamp, keyless auth."""

    def __init__(
        self,
        *,
        halogen_effort: str = DEFAULT_EFFORT,
        halogen_preserve_thinking: bool = False,
        halogen_ctx: int = DEFAULT_CTX,
        halogen_api_key: str = "",
        **kwargs: Any,
    ) -> None:
        # The halogen_* config is ours, not ProviderProfile fields: strip it
        # before the (dataclass) base constructor sees it, keep it as private
        # attributes for the hooks below.
        super().__init__(**kwargs)
        self._halogen_effort = halogen_effort
        self._halogen_preserve_thinking = halogen_preserve_thinking
        self._halogen_ctx = halogen_ctx
        self._halogen_api_key = halogen_api_key

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        return REASONING_EFFORTS

    def default_reasoning_config(self, model: str | None = None) -> dict | None:
        if self._halogen_effort == "off":
            return {"enabled": False}
        return {"enabled": True, "effort": self._halogen_effort}

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **ctx: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Wire fields go in extra_body only (first tuple element) — halogen
        # rejects unknown top-level request fields. Hermes passes extra
        # context kwargs (model, base_url, ...) that this route does not need.
        fields = resolve_thinking(
            reasoning_config,
            default_effort=self._halogen_effort,
            preserve_thinking=self._halogen_preserve_thinking,
        )
        return fields, {}

    def build_client_kwargs_extras(self, **ctx: Any) -> dict[str, Any]:
        # halogen has no auth; Hermes still needs a key-shaped value.
        return {"api_key": self._halogen_api_key or KEYLESS_API_KEY}

    def get_max_tokens(self, model: str | None) -> int | None:
        return clamp_output_budget(self.default_max_tokens, self._halogen_ctx)

    def get_model_context_length(self, model: str) -> int | None:
        return self._halogen_ctx


def build_profile(env: Mapping[str, str]) -> HalogenProfile:
    """Construct the halogen profile from an env mapping (injected for tests).

    Malformed values fall back to defaults — Hermes startup must never crash
    on a bad environment.
    """
    ctx = _env_int(env, "HALOGEN_CTX", DEFAULT_CTX)
    vision = _env_bool(env, "HALOGEN_VISION")
    return HalogenProfile(
        name="halogen",
        aliases=("halo",),
        display_name="Halogen",
        description="Halogen (halogen-flash-server) — Qwen3.8-Flash-Next on Strix Halo",
        env_vars=("HALOGEN_API_KEY", "HALOGEN_BASE_URL"),
        base_url=_env_str(env, "HALOGEN_BASE_URL", DEFAULT_BASE_URL),
        auth_type="api_key",
        supports_vision=vision,
        fallback_models=(MODEL_ID,),
        default_aux_model=MODEL_ID,
        default_max_tokens=_env_int(env, "HALOGEN_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        model_capabilities={
            MODEL_ID: {
                "supports_tools": True,
                "supports_reasoning": True,
                "supports_vision": vision,
                "context_window": ctx,
            },
        },
        classify_api_error=classify_halogen_error,
        halogen_effort=_env_effort(env, "HALOGEN_REASONING_EFFORT", DEFAULT_EFFORT),
        halogen_preserve_thinking=_env_bool(env, "HALOGEN_PRESERVE_THINKING"),
        halogen_ctx=ctx,
        halogen_api_key=_env_str(env, "HALOGEN_API_KEY", ""),
    )


halogen = build_profile(os.environ)
