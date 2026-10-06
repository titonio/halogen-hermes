"""Tests for the halogen error classification module (Task 2).

Covers `classify_halogen_error`: the hook claims ONLY halogen's token-budget 400
(`context_overflow` + the `should_compress` recovery hint Hermes' compaction
reads) and declines everything else, so Hermes' built-in handlers answer the
other statuses with their own, richer hint flags.
Run from the repo root: `.venv/bin/python -m pytest tests/test_errors.py -v`
"""

import pytest

import errors

OVERFLOW = {"reason": "context_overflow", "should_compress": True}

# Verbatim halogen rejection wording (see design spec: halogen admits a
# request only if max_tokens + prompt <= context).
HALOGEN_BUDGET_MESSAGE = (
    "max_tokens 65536 does not fit: prompt is 210000 and the context is "
    "262144, leaving room for 52144"
)


# --- 400 + halogen budget wording -> context_overflow ------------------------


@pytest.mark.parametrize(
    "message",
    [
        HALOGEN_BUDGET_MESSAGE,  # "does not fit" + "leaving room for"
        "prompt is 261000 of 262144, leaving room for 512",  # "leaving room for"
        "prompt exceeds the available context window",  # "context window"
        "the request exceeds the context budget for this model",  # "exceeds the context"
    ],
    ids=["verbatim", "leaving_room_for", "context_window", "exceeds_the_context"],
)
def test_400_budget_wording_is_context_overflow(message):
    assert errors.classify_halogen_error(status_code=400, message=message) == OVERFLOW


def test_400_budget_wording_is_case_insensitive():
    assert errors.classify_halogen_error(
        status_code=400, message="Max_Tokens 65536 Does Not Fit The Context"
    ) == OVERFLOW


# --- 400 without budget wording -> decline (built-in 400 handler decides) -----


@pytest.mark.parametrize(
    "message",
    [
        "invalid role in context",
        "no context configured",
        "unsupported parameter: 'foo'",
        None,
    ],
    ids=["role_in_context", "no_context", "other", "none"],
)
def test_400_without_budget_wording_declines(message):
    assert errors.classify_halogen_error(status_code=400, message=message) is None


def test_400_without_message_kwarg_declines():
    assert errors.classify_halogen_error(status_code=400) is None


# --- every other status -> decline -------------------------------------------


@pytest.mark.parametrize(
    "status_code", [401, 403, 404, 418, 429, 500, 502, 503, None]
)
def test_non_400_statuses_decline_even_with_budget_wording(status_code):
    # Only the 400 branch is claimed: the built-ins map these statuses with the
    # hint flags this hook cannot add (rotation, fallback, no-retry).
    assert (
        errors.classify_halogen_error(
            status_code=status_code, message=HALOGEN_BUDGET_MESSAGE
        )
        is None
    )


def test_no_arguments_declines():
    assert errors.classify_halogen_error() is None


# --- Hermes classify_api_error call convention -------------------------------


def test_full_call_convention_with_extra_kwargs():
    assert errors.classify_halogen_error(
        RuntimeError("Error code: 400"),
        status_code=400,
        error_code="budget_exceeded",
        message=HALOGEN_BUDGET_MESSAGE,
        body={"error": {"code": "budget_exceeded", "message": HALOGEN_BUDGET_MESSAGE}},
        model="halogen-qwen3.8-flash-next",
        headers={"content-length": "128"},
    ) == OVERFLOW
