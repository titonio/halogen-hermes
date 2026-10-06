"""Error classification for the halogen provider profile.

Hermes' ``ProviderProfile.classify_api_error`` hook *replaces* the built-in
verdict rather than refining it: ``_profile_verdict`` feeds the returned dict
straight into ``ClassifiedError(**{**base, **verdict})``, and the dataclass
defaults are ``retryable=True, should_compress=False,
should_rotate_credential=False, should_fallback=False``. A hook that answers a
status with a bare ``{"reason": ...}`` therefore shadows the built-in handler
*and* silently drops the recovery hints it had attached — a strictly weaker
verdict than the one it displaced.

So this hook claims exactly one thing: halogen's token-budget rejection. halogen
admits a request only when ``max_tokens + prompt <= context`` and refuses a
larger one with a 400 in its own wording ("does not fit", "leaving room for",
"context window", "exceeds the context"). Hermes' built-in 400 handler reaches
``context_overflow`` only through generic pattern tables ("max_tokens", "prompt
length", "context window"), so a halogen rejection phrased without any of those
words — "…leaving room for 512 tokens" — lands on ``format_error`` and aborts
the turn instead of compacting. Matching the provider's own wording makes the
verdict independent of that incidental overlap, and it carries
``should_compress=True``, the hint the built-in ``context_overflow`` verdict
sets and the one the compaction recovery reads.

Everything else returns ``None`` and is left to Hermes' built-ins, which answer
401/403 → auth, 404 → model_not_found, 429 → rate_limit (reset-window and
billing nuance included) and 5xx → server_error/overloaded — each with the
correct hint flags (credential rotation, fallback, no-retry), which a bare
reason here could only lose. Unlisted statuses and unlisted 5xx deliberately
decline too: the built-in already maps 5xx to ``server_error``, so claiming them
would add nothing and cost the hints.

Signature matches Hermes' ``classify_api_error`` profile-field call convention;
unexpected extra keyword arguments are absorbed by ``**_ignored``.

Stdlib only.
"""

# halogen's budget-rejection wording, matched on the lowercased message (so the
# markers are lowercase). Deliberately narrow: a bare "context" would also match
# unrelated 400s — "invalid role in context", a WAF page naming a context path —
# and trigger a compaction that cannot fix them.
_CONTEXT_MARKERS: tuple[str, ...] = (
    "does not fit",
    "leaving room for",
    "context window",
    "exceeds the context",
)


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
    """Claim halogen's 400 token-budget rejection; decline everything else.

    Returns ``{"reason": "context_overflow", "should_compress": True}`` only for
    a 400 whose lowercased ``message`` matches a budget marker; ``None`` (Hermes'
    built-ins decide) for every other status, a 400 without budget wording, and
    a missing message. Only ``status_code`` and ``message`` drive the decision;
    the remaining named parameters exist for Hermes' call convention.
    """
    if status_code != 400:
        return None
    text = (message or "").lower()
    if any(marker in text for marker in _CONTEXT_MARKERS):
        return {"reason": "context_overflow", "should_compress": True}
    return None
