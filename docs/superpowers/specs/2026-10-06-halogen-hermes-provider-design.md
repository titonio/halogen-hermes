# Hermes Halogen Provider Plugin — Design

Date: 2026-10-06
Status: approved (conversational design approval; this document is the written spec)

## Goal

Build a Hermes Agent plugin that lets Hermes route its inference through a
[halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)
instance — the OpenAI-compatible server that runs Qwen3.8-Flash-Next at native
speed on AMD Strix Halo. The capability target is the DSH (DeepSeek Harness)
`llm-halogen` adapter installed on this machine at
`/var/lib/dsh/profiles/web/plugins/llm-halogen`, translated onto Hermes'
model-provider plugin surface.

Success looks like: after dropping one directory into `$HERMES_HOME/plugins/model-providers/halogen/`,
`hermes model` lists a `halogen` provider, `hermes doctor` health-checks it,
and a normal Hermes chat/tool-call turn runs against the halogen endpoint with
thinking control working from Hermes' side (which it does not by default for
custom OpenAI-compatible endpoints).

## Background facts (verified on this machine, 2026-10-06)

- Live halogen instance: `http://192.168.31.7:8731` — API/engine 0.16.2,
  model `halogen-qwen3.8-flash-next`, context 262,144, `max_tokens_default`
  8192, `max_tokens_cap` 65536, vision tower **enabled**, tool-call wire format
  Qwen-XML, thinking control probed OK.
- halogen serves `/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
  `/v1/responses`, `/v1/models`, plus `/health` (capability manifest) and
  `/metrics`. No authentication on the API port.
- halogen reads thinking controls from several field shapes, including the
  OpenRouter-style `reasoning` object and the Qwen fields `enable_thinking`,
  `reasoning_effort`, `max_thinking_tokens`.
- Hermes is **not installed** on this host and must not be; verification runs in
  the official `nousresearch/hermes-agent` Docker image (throwaway container).
- Hermes model provider plugins: a directory with `__init__.py` calling
  `register_provider(ProviderProfile(...))` and a `plugin.yaml` declaring
  `kind: model-provider`. Discovery from
  `$HERMES_HOME/plugins/model-providers/<name>/` (drop-in) and from
  `$HERMES_HOME/plugins/<name>/` when installed via `hermes plugins install`.
  Auto-wires `--provider`, `/model` picker, `hermes doctor`, credential
  resolution, and the `chat_completions` transport.

## Reference capability map (DSH `llm-halogen` → Hermes)

| DSH adapter capability | Hermes plugin mechanism |
| --- | --- |
| `halogen` provider route; model id pass-through | `ProviderProfile(name="halogen")`, `fallback_models` |
| `baseURL` default `http://127.0.0.1:8731`, `$HALOGEN_BASE_URL` override | `base_url` + `env_vars=("HALOGEN_API_KEY", "HALOGEN_BASE_URL")` |
| thinking: `enable_thinking` + `reasoning_effort` (`off` → `enable_thinking: false`) | `build_api_kwargs_extras()` / `build_extra_body()` mapping Hermes' reasoning config onto the wire |
| `preserveThinking` → `preserve_thinking: true` (reasoning passback, cache reuse) | env `HALOGEN_PRESERVE_THINKING=1` → same wire field |
| `maxTokens` default 8192; output-budget clamp `min(requested, max(1024, 25% × ctx))` so `max_tokens + prompt ≤ context` holds | `default_max_tokens` + ported clamp in the build hooks |
| catalog `halogen-qwen3.8-flash-next`, ctx 262,144; live `/v1/models` fetch | `fallback_models` + default `fetch_models()`; `model_capabilities` declares tools/reasoning/context |
| vision gated on catalog capability; images as base64 `data:` URLs | `model_capabilities.supports_vision` (env-gated, see Config) |
| error taxonomy (AUTH, RATE_LIMIT, CONTEXT_WINDOW_EXCEEDED, SERVER, …) | `classify_api_error` hook |
| retry policy, stream-idle timeout | **not ported** — Hermes core owns retries/fallbacks |
| `purpose: session-title` forces thinking off | **not ported** — no Hermes seam exposes call purpose to the profile; documented limitation |

## Architecture

One plugin directory, importable standalone for tests, that registers a single
provider profile. No tools, no hooks, no gateway surface — inference routing
only.

```
Halogen-hermes/
├── plugin.yaml       # name: halogen, kind: model-provider, metadata
├── __init__.py       # module-level register_provider(HalogenProfile())
├── profile.py        # HalogenProfile(ProviderProfile) + build hooks
├── wire.py           # effort→wire mapping + output-budget clamp (pure functions)
├── errors.py         # classify_api_error mapping (pure functions)
├── tests/            # pytest; stub `providers` package injected via conftest
├── README.md         # install paths, config reference, zero-code fallback
└── docs/superpowers/specs/…
```

- `wire.py` and `errors.py` are pure stdlib modules with no Hermes imports —
  they carry all the interesting logic and are unit-testable as-is.
- `profile.py` imports `providers.base.ProviderProfile` lazily enough that the
  test conftest can inject a stub `providers` package before import.
- `__init__.py` does the registration call at module level, matching Hermes'
  discovery contract.

## Wire mapping (the core logic)

Effort vocabulary: `off | minimal | low | medium | high | xhigh` (halogen's
`/health` depths; Hermes' reasoning-effort names fold onto these — Hermes'
vocabulary is checked against this list at implementation time and unknown
levels fall back to the configured default rather than erroring).

| Resolved effort | Wire fields |
| --- | --- |
| `off` | `enable_thinking: false` |
| `minimal`…`xhigh` | `enable_thinking: true`, `reasoning_effort: <level>` |
| unset (no Hermes reasoning config) | configured default effort applies (default `medium`) |
| any + `HALOGEN_PRESERVE_THINKING=1` | additionally `preserve_thinking: true` |

Output budget: when the request carries an output cap, send
`max_completion_tokens = min(cap, max(1024, floor(0.25 × context_window)))`.
The clamp mirrors the DSH adapter: halogen admits a request only if
`max_tokens + prompt ≤ context`, and the client cannot tokenize the prompt.

Assistant history replay (`reasoning_content` passback) is handled by Hermes'
chat_completions transport when the provider declares reasoning support; the
plugin does not re-serialize messages.

## Config surface (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `HALOGEN_BASE_URL` | `http://127.0.0.1:8731/v1` | endpoint base |
| `HALOGEN_API_KEY` | unset (halogen has no auth) | optional bearer token for proxied deployments |
| `HALOGEN_REASONING_EFFORT` | `medium` | default effort when Hermes sends none |
| `HALOGEN_PRESERVE_THINKING` | unset | `1` adds `preserve_thinking: true` |
| `HALOGEN_MAX_TOKENS` | `8192` | provider-level output cap (server cap is 65536) |
| `HALOGEN_CTX` | `262144` | declared context window (YaRN deployments raise it) |
| `HALOGEN_VISION` | unset | `1` declares `supports_vision: true` for the catalog model |

All read at profile construction (process start). Hermes re-discovers provider
profiles per process; changing env means restarting Hermes — same posture as
the DSH adapter's "next request picks up settings" minus the live-reload seam,
which Hermes does not offer to provider profiles.

## Error handling

`classify_api_error` maps, by HTTP status and message shape:

- 401/403 → auth failure
- 429 → rate limit
- 400 whose message matches halogen's token-budget/context wording ("does not
  fit", "leaving room for", budget-cap phrasing) → context-window-exceeded so
  Hermes' compaction can recover
- other 400 → invalid request
- 5xx → server error

Exact Hermes error class names are taken from `providers/base.py` at
implementation time; the mapping table lives in `errors.py` as pure data.

## Vision

The halogen vision tower is off by default per deployment; the plugin declares
`supports_vision` only when `HALOGEN_VISION=1`, matching the live instance's
`/health` (`vision.enabled: true`). Image content parts pass through Hermes'
standard OpenAI image shape (base64 `data:` URLs — halogen refuses http(s)
image URLs by design). Image-token estimation is not ported; Hermes' own
context accounting governs.

## Testing

1. **Unit (pytest, no Hermes installed):** wire mapping table (every effort
   level, off, preserve_thinking, default precedence), output-budget clamp
   edges (tiny ctx, cap below floor, no cap), error classification table, env
   override parsing, profile field assertions against a stubbed
   `providers.base.ProviderProfile`.
2. **Wire smoke (live halogen at 192.168.31.7:8731):** plain chat, thinking
   on/off + effort, tool-call round-trip, `/v1/models` shape — via a small
   script, no Hermes needed.
3. **End-to-end (throwaway Docker):** `nousresearch/hermes-agent` image with
   the plugin bind-mounted at `/opt/data/plugins/model-providers/halogen/`;
   verify `hermes plugins doctor --ci` (or `validate`), provider discovery in
   `hermes model`/`hermes doctor`, and one real chat + tool-call turn with
   `--provider halogen -m halogen-qwen3.8-flash-next`. Container and its data
   dir are deleted afterwards; nothing persists on this host user.

## Distribution

- Repo root **is** the plugin directory (repo root carries `plugin.yaml` +
  `__init__.py`), so all three paths work:
  1. drop-in copy to `~/.hermes/plugins/model-providers/halogen/`
  2. `hermes plugins install <owner>/halogen-hermes`
  3. Docker bind-mount (verification layout)
- README documents all three plus the zero-code fallback (a `custom_providers`
  config block pointing at halogen with server-side `HALOGEN_*` sampling
  defaults) for users who want no plugin at all.
- No Python dependencies beyond the standard library (Hermes ships `openai`
  already).

## Out of scope

- Retry policy / stream-idle timeouts (Hermes core owns these).
- `tool_choice` mapping (not in the DSH reference either).
- Classifier/logprobs tooling, `/health`-probing tools, image-token estimation.
- `/v1/responses` or `/v1/messages` api_modes — `chat_completions` is the
  Hermes transport we target.
- Catalog submission to the Hermes plugin catalog (possible follow-up; its
  admission checklist is a separate task).

## Known limitations (documented in README)

- Call-purpose awareness (DSH forces thinking off for session titles) has no
  Hermes provider-profile seam; compaction summaries run at the configured
  effort.
- Env-var config is read at process start; no live reload.
- `supports_vision` is a static declaration, not probed from `/health`.
