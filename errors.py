"""Error classification for the halogen provider profile.

Maps halogen-flash-server failures to Hermes failover reasons. The returned
``{"reason": ...}`` values are Hermes ``FailoverReason`` names: ``auth``,
``model_not_found``, ``rate_limit``, ``context_overflow``, ``format_error``,
``overloaded``, ``server_error``. Returning ``None`` declines the error and
lets Hermes' built-in classification decide.

The operationally important case is a 400 whose message carries halogen's
token-budget wording — ``max_tokens 65536 does not fit: prompt is 210000 and
the context is 262144, leaving room for 52144``. Hermes must see
``context_overflow`` (triggers compaction recovery), not ``format_error``
(aborts the turn).

Signature matches Hermes' ``classify_api_error`` profile-field call
convention; unexpected extra keyword arguments are absorbed by ``**_ignored``.

Stdlib only.
"""

# halogen rejects an over-budget request with wording along these lines; the
# match is on the lowercased message, so these markers are lowercase too.
_CONTEXT_MARKERS: tuple[str, ...] = ("does not fit", "leaving room for", "context")

# Statuses halogen is known to emit, mapped to Hermes failover reasons.
# Anything absent (including other 5xx) declines to Hermes' built-ins.
_STATUS_REASONS: dict[int, str] = {
    401: "auth",
    403: "auth",
    404: "model_not_found",
    429: "rate_limit",
    500: "server_error",
    502: "server_error",
    503: "overloaded",
}


def classify_halogen_error(
    error: object = None,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    message: str | None = None,
    body: object = None,
    model: str | None = None,
    **_ignored: object,
) -> dict | None:
    """Classify a halogen API error as a Hermes failover reason.

    Only ``status_code`` and ``message`` drive the decision; the remaining
    named parameters exist for Hermes' call convention. A 400 counts as
    ``context_overflow`` when its lowercased message mentions halogen's
    budget/context wording, otherwise ``format_error``; a missing message can
    not match, so it is a ``format_error`` too.
    """
    if status_code is None:
        return None
    if status_code == 400:
        text = (message or "").lower()
        if any(marker in text for marker in _CONTEXT_MARKERS):
            return {"reason": "context_overflow"}
        return {"reason": "format_error"}
    reason = _STATUS_REASONS.get(status_code)
    return {"reason": reason} if reason is not None else None
