"""Wire mapping between Hermes reasoning config and halogen-flash-server.

halogen-flash-server is an OpenAI-compatible Qwen inference server whose
thinking controls travel in the request body as `enable_thinking`,
`reasoning_effort`, and `preserve_thinking`. The dict returned by
`resolve_thinking` is intended to be merged into the OpenAI SDK `extra_body`.

Stdlib only.
"""

EFFORT_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")

# Hermes effort ladder, weakest to strongest. EFFORT_LEVELS is the contiguous
# middle slice; "max"/"ultra" sit above the wire vocabulary and clamp down.
_LADDER: tuple[str, ...] = ("none",) + EFFORT_LEVELS + ("max", "ultra")


def _clamp_effort(effort: str, default_effort: str) -> str:
    """Nearest level in EFFORT_LEVELS at or below `effort`.

    The only ladder levels above the wire vocabulary are "max" and "ultra"
    ("none" and "" are handled by the caller before clamping), so both clamp
    to "xhigh". Unknown strings fall back to `default_effort`, which may
    itself be "off".
    """
    if effort in EFFORT_LEVELS:
        return effort
    if effort in _LADDER:
        return EFFORT_LEVELS[-1]
    return default_effort


def _off(preserve_thinking: bool) -> dict:
    out = {"enable_thinking": False}
    if preserve_thinking:
        out["preserve_thinking"] = True
    return out


def resolve_thinking(
    reasoning_config: dict | None,
    *,
    default_effort: str = "medium",
    preserve_thinking: bool = False,
) -> dict:
    """Map a Hermes reasoning config to halogen wire fields for extra_body.

    Returns either ``{"enable_thinking": False}`` or
    ``{"enable_thinking": True, "reasoning_effort": <level>}``, plus
    ``"preserve_thinking": True`` in both branches when configured (mirrors
    the DSH llm-halogen reference adapter). ``enabled is False`` or effort
    ``"none"`` means off; an unset effort applies ``default_effort`` (which
    may be ``"off"``); ``max``/``ultra`` clamp to ``xhigh``; unknown effort
    strings fall back to ``default_effort``.
    """
    config = reasoning_config or {}
    effort = str(config.get("effort") or "").strip().lower()
    if config.get("enabled") is False or effort == "none":
        return _off(preserve_thinking)
    resolved = default_effort if effort == "" else _clamp_effort(effort, default_effort)
    if resolved == "off":
        return _off(preserve_thinking)
    out = {"enable_thinking": True, "reasoning_effort": resolved}
    if preserve_thinking:
        out["preserve_thinking"] = True
    return out


def clamp_output_budget(max_tokens: int | None, context_window: int | None) -> int | None:
    """Clamp a requested output budget to a quarter of the context window.

    Result is ``min(max_tokens, max(1024, floor(0.25 * context_window)))``:
    a cap below the 1024 floor is preserved as-is. ``None`` (no cap) passes
    through; a missing or non-positive context window returns the cap
    unchanged.
    """
    if max_tokens is None:
        return None
    if not context_window or context_window <= 0:
        return max_tokens
    return min(max_tokens, max(1024, context_window // 4))
