"""Tests for the halogen wire mapping module (Task 1).

Covers `resolve_thinking` (Hermes reasoning config -> halogen-flash-server
extra_body wire fields) and `clamp_output_budget` (output token budget clamp).
Run from the repo root: `.venv/bin/python -m pytest tests/test_wire.py -v`
"""

import wire


# --- EFFORT_LEVELS ---------------------------------------------------------


def test_effort_levels_tuple():
    assert wire.EFFORT_LEVELS == ("minimal", "low", "medium", "high", "xhigh")


# --- resolve_thinking: on/off mapping ---------------------------------------


def test_enabled_low_effort():
    assert wire.resolve_thinking({"enabled": True, "effort": "low"}) == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }


def test_disabled_is_off():
    assert wire.resolve_thinking({"enabled": False}) == {"enable_thinking": False}


def test_effort_none_is_off():
    assert wire.resolve_thinking({"effort": "none"}) == {"enable_thinking": False}


def test_none_config_uses_default_medium_on():
    assert wire.resolve_thinking(None) == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


def test_empty_effort_uses_default_on():
    assert wire.resolve_thinking({"effort": ""}) == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


def test_empty_config_uses_default_on():
    assert wire.resolve_thinking({}) == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


def test_enabled_false_beats_explicit_effort():
    # `enabled` is the switch: an effort sitting next to it does not turn
    # thinking back on.
    assert wire.resolve_thinking({"enabled": False, "effort": "low"}) == {
        "enable_thinking": False
    }


# --- resolve_thinking: clamping and fallback --------------------------------


def test_effort_max_clamps_to_xhigh():
    assert wire.resolve_thinking({"effort": "max"}) == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }


def test_effort_ultra_clamps_to_xhigh():
    assert wire.resolve_thinking({"effort": "ultra"}) == {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    }


def test_unknown_effort_falls_back_to_default():
    assert wire.resolve_thinking({"effort": "bogus"}) == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


def test_default_effort_off_with_no_config():
    assert wire.resolve_thinking(None, default_effort="off") == {
        "enable_thinking": False
    }


# --- resolve_thinking: preserve_thinking in both branches --------------------


def test_preserve_thinking_on_branch():
    assert wire.resolve_thinking(
        {"enabled": True, "effort": "low"}, preserve_thinking=True
    ) == {
        "enable_thinking": True,
        "reasoning_effort": "low",
        "preserve_thinking": True,
    }


def test_preserve_thinking_off_branch():
    assert wire.resolve_thinking({"enabled": False}, preserve_thinking=True) == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }


# --- clamp_output_budget ------------------------------------------------------


def test_clamp_cap_below_quarter_ctx_unchanged():
    assert wire.clamp_output_budget(8192, 262144) == 8192


def test_clamp_caps_at_quarter_ctx():
    assert wire.clamp_output_budget(100000, 262144) == 65536


def test_clamp_small_cap_preserved():
    assert wire.clamp_output_budget(500, 262144) == 500


def test_clamp_missing_ctx_returns_cap():
    assert wire.clamp_output_budget(100000, None) == 100000


def test_clamp_none_cap_passes_through():
    assert wire.clamp_output_budget(None, 262144) is None


def test_clamp_floor_is_1024():
    assert wire.clamp_output_budget(100000, 2048) == 1024


def test_clamp_nonpositive_ctx_returns_cap():
    assert wire.clamp_output_budget(100000, 0) == 100000
