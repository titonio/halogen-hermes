"""Tests for the halogen error classification module (Task 2).

Covers `classify_halogen_error`: HTTP status -> Hermes FailoverReason name,
including halogen's token-budget/context wording on 400 responses, which must
classify as `context_overflow` so Hermes' compaction can recover instead of
aborting on a `format_error`.
Run from the repo root: `.venv/bin/python -m pytest tests/test_errors.py -v`
"""

import errors


# --- status-only mapping -----------------------------------------------------


def test_401_is_auth():
    assert errors.classify_halogen_error(status_code=401) == {"reason": "auth"}


def test_403_is_auth():
    assert errors.classify_halogen_error(status_code=403) == {"reason": "auth"}


def test_404_is_model_not_found():
    assert errors.classify_halogen_error(status_code=404) == {"reason": "model_not_found"}


def test_429_is_rate_limit():
    assert errors.classify_halogen_error(status_code=429) == {"reason": "rate_limit"}


def test_503_is_overloaded():
    assert errors.classify_halogen_error(status_code=503) == {"reason": "overloaded"}


def test_500_is_server_error():
    assert errors.classify_halogen_error(status_code=500) == {"reason": "server_error"}


def test_502_is_server_error():
    assert errors.classify_halogen_error(status_code=502) == {"reason": "server_error"}


# --- 400: halogen context/budget wording vs generic format error --------------


# Verbatim halogen rejection wording (see design spec: halogen admits a
# request only if max_tokens + prompt <= context).
HALOGEN_BUDGET_MESSAGE = (
    "max_tokens 65536 does not fit: prompt is 210000 and the context is "
    "262144, leaving room for 52144"
)


def test_400_halogen_budget_wording_is_context_overflow():
    assert errors.classify_halogen_error(
        status_code=400, message=HALOGEN_BUDGET_MESSAGE
    ) == {"reason": "context_overflow"}


def test_400_exceeds_the_context_wording_is_context_overflow():
    assert errors.classify_halogen_error(
        status_code=400,
        message="prompt length 210000 exceeds the context window of 262144",
    ) == {"reason": "context_overflow"}


def test_400_context_wording_is_case_insensitive():
    assert errors.classify_halogen_error(
        status_code=400, message="Max_Tokens 65536 Does Not Fit The Context"
    ) == {"reason": "context_overflow"}


def test_400_other_message_is_format_error():
    assert errors.classify_halogen_error(
        status_code=400, message="invalid role: 'systemx'"
    ) == {"reason": "format_error"}


def test_400_message_none_is_format_error():
    assert errors.classify_halogen_error(status_code=400, message=None) == {
        "reason": "format_error"
    }


def test_400_without_message_is_format_error():
    assert errors.classify_halogen_error(status_code=400) == {"reason": "format_error"}


# --- declines (let Hermes' built-ins decide) ---------------------------------


def test_status_none_declines():
    assert errors.classify_halogen_error(status_code=None) is None


def test_no_arguments_declines():
    assert errors.classify_halogen_error() is None


def test_unknown_status_declines():
    assert errors.classify_halogen_error(status_code=418) is None


# --- Hermes classify_api_error call convention -------------------------------


def test_full_call_convention_with_extra_kwargs():
    assert errors.classify_halogen_error(
        RuntimeError("Error code: 429"),
        status_code=429,
        error_code="rate_limited",
        message="Too Many Requests",
        body={"error": {"code": "rate_limited", "message": "Too Many Requests"}},
        model="halogen-qwen3.8-flash-next",
        headers={"retry-after": "2"},
    ) == {"reason": "rate_limit"}
