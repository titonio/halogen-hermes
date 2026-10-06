# Hermes Halogen Provider Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Hermes Agent model-provider plugin that routes Hermes inference through a halogen-flash-server instance with working thinking control, tool calls, vision gating, and error classification.

**Architecture:** One Hermes model-provider plugin directory (repo root is the plugin dir): a `ProviderProfile` subclass registered at import, with all decision logic in two pure-stdlib modules (`wire.py`, `errors.py`) that are unit-tested without Hermes installed. Verification: pytest units → direct wire smoke against the live halogen → throwaway `nousresearch/hermes-agent` Docker container with the plugin bind-mounted.

**Tech Stack:** Python 3.10+ stdlib only (plugin code); pytest; Docker (podman also fine) for e2e; live halogen-flash-server at `http://192.168.31.7:8731`.

**Spec:** `docs/superpowers/specs/2026-10-06-halogen-hermes-provider-design.md`

## Global Constraints

- Plugin runtime code imports **stdlib only** (no third-party imports in `wire.py`, `errors.py`, `profile.py`, `__init__.py` beyond `providers`/`providers.base` which Hermes provides).
- Wire field names verbatim: `enable_thinking`, `reasoning_effort`, `preserve_thinking` — sent via **extra_body** (the openai SDK merges extra_body into the JSON body; unknown top-level kwargs would be rejected by the SDK).
- Halogen effort vocabulary: `minimal | low | medium | high | xhigh`; Hermes `none`/disabled → `enable_thinking: false`; Hermes `max`/`ultra` clamp to `xhigh` (nearest weaker).
- Defaults: effort `medium`, `max_tokens` 8192, context window 262144, base URL `http://127.0.0.1:8731/v1`, model id `halogen-qwen3.8-flash-next`.
- Env vars (read at profile construction): `HALOGEN_BASE_URL`, `HALOGEN_API_KEY`, `HALOGEN_REASONING_EFFORT`, `HALOGEN_PRESERVE_THINKING`, `HALOGEN_MAX_TOKENS`, `HALOGEN_CTX`, `HALOGEN_VISION`.
- Hermes facts verified from source (`/tmp/hermes-src`, clone of NousResearch/hermes-agent main): `ProviderProfile` is a dataclass in `providers.base`; `register_provider(profile)` from `providers`; `reasoning_config` dict shape `{"enabled": bool, "effort": str}`; `classify_api_error` is a profile **field** `Callable[..., Any] | None` called as `(error, *, status_code, error_code, message, body, model) -> Mapping | None` returning `{"reason": <FailoverReason name>}`; FailoverReason names used here: `auth`, `model_not_found`, `rate_limit`, `context_overflow`, `format_error`, `overloaded`, `server_error`; keyless placeholder api_key convention is `no-key-required`; one-shot CLI is `hermes -q "<prompt>"`; user plugin drop-in path is `$HERMES_HOME/plugins/model-providers/<dir>/` with `plugin.yaml` (`kind: model-provider`) + `__init__.py`.
- Commit after every task; conventional messages (`feat:`, `test:`, `docs:`, `chore:`).

## Review Focus

Failure modes the spec implies that task tests must pin explicitly:

1. Hermes effort levels outside halogen's set (`max`, `ultra`) must clamp to `xhigh`, never reach the wire verbatim (halogen 400s unknown levels).
2. Degenerate `reasoning_config` shapes (`None`, `{}`, `{"effort": ""}`, `{"enabled": false}`) must map deterministically (default-effort / off) without raising.
3. Malformed env values (`HALOGEN_REASONING_EFFORT=bogus`, `HALOGEN_CTX=abc`) must fall back to defaults at import, never crash Hermes startup.
4. halogen's token-budget 400 wording ("does not fit", "leaving room for") must classify as `context_overflow` (compaction recovery), not `format_error` (abort).
5. Client construction without any credential must still yield a usable client (`no-key-required` placeholder default).

---

### Task 1: Wire mapping module (`wire.py`)

**Files:**
- Create: `wire.py`
- Test: `tests/test_wire.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces:
  - `EFFORT_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")`
  - `resolve_thinking(reasoning_config: dict | None, *, default_effort: str = "medium", preserve_thinking: bool = False) -> dict` → extra_body dict, one of: `{"enable_thinking": False}` (+`preserve_thinking`), or `{"enable_thinking": True, "reasoning_effort": <level>}` (+`preserve_thinking`). `default_effort` may be `"off"`.
  - `clamp_output_budget(max_tokens: int | None, context_window: int | None) -> int | None` → `min(cap, max(1024, floor(0.25 × ctx)))`; `None` passes through; missing/≤0 ctx returns cap unchanged.

- [ ] **Step 1: Write failing tests** in `tests/test_wire.py` covering: `{"enabled": True, "effort": "low"}` → `{"enable_thinking": True, "reasoning_effort": "low"}`; `{"enabled": False}` → `{"enable_thinking": False}`; `{"effort": "none"}` → `{"enable_thinking": False}`; `None` → default effort medium on; `{"effort": ""}` → default effort on; `{"effort": "max"}` and `{"effort": "ultra"}` → `xhigh`; `{"effort": "bogus"}` → falls back to default effort (Review Focus 1–2); `default_effort="off"` with no config → `{"enable_thinking": False}`; `preserve_thinking=True` adds `"preserve_thinking": True` in both the on and off branches (mirrors DSH reference); clamp edges: `(8192, 262144)→8192`, `(100000, 262144)→65536`, `(500, 262144)→500`, `(100000, None)→100000`, `(None, 262144)→None`, `(100000, 2048)→1024`.

- [ ] **Step 2: Run, verify they fail** — `python3 -m pytest tests/test_wire.py -v` → collection error / ImportError.

- [ ] **Step 3: Implement `wire.py`.** `resolve_thinking`: normalize `effort = str((reasoning_config or {}).get("effort") or "").strip().lower()`; treat `enabled is False` or `effort in ("", "none")` per the table above (empty effort means "unset" → default effort applies; `enabled False` means off). Clamp ladder: Hermes ladder order is `none < minimal < low < medium < high < xhigh < max < ultra`; clamp = nearest level in `EFFORT_LEVELS` at-or-below the requested one (so `max`/`ultra`→`xhigh`, unknown strings → `default_effort`). Off branch still carries `preserve_thinking` when configured.

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit** `feat: halogen wire mapping (thinking fields + output budget clamp)`

---

### Task 2: Error classification module (`errors.py`)

**Files:**
- Create: `errors.py`
- Test: `tests/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `classify_halogen_error(error: object = None, *, status_code: int | None = None, error_code: str | None = None, message: str | None = None, body: object = None, model: str | None = None) -> dict | None` — signature matches Hermes' `classify_api_error` profile field (extra kwargs tolerated via the named params + `**_ignored` if needed). Returns `{"reason": "<FailoverReason name>"}` or `None` to decline.

- [ ] **Step 1: Write failing tests** in `tests/test_errors.py`: 401/403 → `auth`; 404 → `model_not_found`; 429 → `rate_limit`; 400 + message `"max_tokens 65536 does not fit: prompt is 210000 and the context is 262144, leaving room for 52144"` → `context_overflow` (Review Focus 4); 400 + message containing `"exceeds the context"` → `context_overflow`; 400 other → `format_error`; 503 → `overloaded`; 500/502 → `server_error`; status None → `None`; 418 → `None` (decline, let Hermes built-ins decide).

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `errors.py`** as a small ordered rule table (status → reason), with the 400 branch inspecting lowercased `message` for `"does not fit"` / `"leaving room for"` / `"context"` before defaulting to `format_error`.

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit** `feat: halogen error classification for Hermes failover reasons`

---

### Task 3: Provider profile + plugin registration

**Files:**
- Create: `profile.py`, `__init__.py`, `plugin.yaml`, `tests/conftest.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: `wire.resolve_thinking`, `wire.clamp_output_budget`, `wire.EFFORT_LEVELS`, `errors.classify_halogen_error`.
- Produces: module-level `halogen` profile instance; `__init__.py` calls `providers.register_provider(halogen)` at import. `profile.py` exposes `build_profile(env: Mapping[str, str]) -> ProviderProfile` (env injected for testability) and `HalogenProfile(ProviderProfile)` subclass.

- [ ] **Step 1: Write `tests/conftest.py`** — before importing `profile`, inject stub modules: `providers` (with `register_provider(profile)` appending to `providers.REGISTRY: dict[str, object]`) and `providers.base` (with a `ProviderProfile` class whose `__init__(**kwargs)` sets each kwarg as an attribute, and no-op method defaults for `get_max_tokens`/`get_model_context_length` returning `getattr(self, ...)` defaults). Guard: skip injection if the real `providers` package imports.

- [ ] **Step 2: Write failing tests** in `tests/test_profile.py` (import `profile` after conftest stub): default env → `name == "halogen"`, `base_url == "http://127.0.0.1:8731/v1"`, `fallback_models == ("halogen-qwen3.8-flash-next",)`, `default_aux_model == "halogen-qwen3.8-flash-next"`, `default_max_tokens == 8192`, `model_capabilities["halogen-qwen3.8-flash-next"] == {"supports_tools": True, "supports_reasoning": True, "supports_vision": False, "context_window": 262144}`; `HALOGEN_BASE_URL`/`HALOGEN_CTX=1048576`/`HALOGEN_VISION=1`/`HALOGEN_MAX_TOKENS=65536` overrides land; `HALOGEN_REASONING_EFFORT=bogus` and `HALOGEN_CTX=abc` fall back to defaults (Review Focus 3); `supported_reasoning_efforts(None) == ("none", "minimal", "low", "medium", "high", "xhigh")`; `default_reasoning_config()` → `{"enabled": True, "effort": "medium"}`, and `{"enabled": False}` when env effort is `off`; `build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"})` → `({"enable_thinking": True, "reasoning_effort": "high"}, {})` (extra_body side, Review Focus: nothing top-level); `build_client_kwargs_extras()` → `{"api_key": "no-key-required"}` (Review Focus 5); `get_max_tokens(None)` with env `HALOGEN_MAX_TOKENS=100000` → `65536` (clamp applied); `get_model_context_length("anything")` → declared ctx; `profile.classify_api_error is classify_halogen_error`; `__init__.py` import registers under `"halogen"` in the stub registry.

- [ ] **Step 3: Run, verify fail.**

- [ ] **Step 4: Implement `profile.py` + `__init__.py` + `plugin.yaml`.** `build_profile(env)` reads the seven env vars with safe parsers (`_env_int` falls back on ValueError; effort validated against `EFFORT_LEVELS + ("off",)`, fallback `medium`; bools are `== "1"`). `HalogenProfile` overrides: `supported_reasoning_efforts` (the tuple above), `default_reasoning_config`, `build_api_kwargs_extras` (delegates to `wire.resolve_thinking`, returns `(fields, {})`), `get_max_tokens` (clamp via `wire.clamp_output_budget(self.default_max_tokens, self._halogen_ctx)`), `get_model_context_length`, `build_client_kwargs_extras`. `plugin.yaml`: `name: halogen-provider`, `kind: model-provider`, `version: 0.1.0`, `description: Halogen (halogen-flash-server) — Qwen3.8-Flash-Next on Strix Halo`, `author:` (ask user or use placeholder from git config). `__init__.py`: import profile, call `register_provider`, re-export `halogen`.

- [ ] **Step 5: Run tests, verify pass.**

- [ ] **Step 6: Commit** `feat: halogen provider profile with thinking, budget, and auth defaults`

---

### Task 4: Live wire smoke script

**Files:**
- Create: `scripts/wire_smoke.py`

**Interfaces:**
- Consumes: `wire.resolve_thinking` (import from repo root).
- Produces: exit-0/1 script proving the wire contract against a real halogen; used by Task 5 and by users.

- [ ] **Step 1: Write `scripts/wire_smoke.py`** (stdlib `urllib`): base URL from `HALOGEN_BASE_URL` (default `http://127.0.0.1:8731/v1`); checks — (a) `GET {base}/models` lists `halogen-qwen3.8-flash-next`; (b) chat completion with `messages=[{"role":"user","content":"Reply with exactly: HALOGEN_SMOKE_OK"}]`, `stream: false`, plus `resolve_thinking({"enabled": True, "effort": "minimal"})` merged into the body → 200 and non-empty `choices[0].message.content`; (c) same with `{"enabled": False}` → 200; (d) tool round-trip: declare one trivial `get_weather` tool, ask "What's the weather in Lisbon?", assert `choices[0].message.tool_calls` non-empty. Print one line per check; nonzero exit on any failure.

- [ ] **Step 2: Run against the live instance** — `HALOGEN_BASE_URL=http://192.168.31.7:8731/v1 python3 scripts/wire_smoke.py` → all four checks PASS.

- [ ] **Step 3: Commit** `feat: live wire smoke script for halogen endpoints`

---

### Task 5: README + Docker end-to-end verification

**Files:**
- Create: `README.md`, `scripts/docker_verify.sh`

**Interfaces:**
- Consumes: whole plugin dir; `scripts/wire_smoke.py`.
- Produces: throwaway e2e proof; user-facing docs.

- [ ] **Step 1: Write `scripts/docker_verify.sh`** — `set -euo pipefail`; `IMAGE=${IMAGE:-nousresearch/hermes-agent:latest}`; `DATA=$(mktemp -d)`; copy plugin files (everything except `tests/`, `scripts/`, `docs/`, `.git`) into `$DATA/plugins/model-providers/halogen/`; write `$DATA/config.yaml` with `model: {provider: halogen, name: halogen-qwen3.8-flash-next}`; `docker run --rm -e HALOGEN_BASE_URL -v "$DATA:/opt/data" $IMAGE hermes plugins list` (halogen must appear — if the image lacks this subcommand, fall back to `hermes doctor`); `docker run ... hermes -q "Reply with exactly: HALOGEN_E2E_OK"` must print the token; trap-removes `$DATA`. If the image's config schema rejects `model.provider: halogen`, adjust to the schema the container reports and note it in the README.

- [ ] **Step 2: Run it** — `HALOGEN_BASE_URL=http://192.168.31.7:8731/v1 bash scripts/docker_verify.sh` → plugin listed, `HALOGEN_E2E_OK` printed. Nothing persists on the host beyond the temp dir.

- [ ] **Step 3: Write `README.md`** — what it is; three install paths (drop-in copy to `~/.hermes/plugins/model-providers/halogen/`, `hermes plugins install <owner>/halogen-hermes`, Docker bind-mount as in `docker_verify.sh`); env-var config table (copy from spec); `scripts/wire_smoke.py` usage; zero-code fallback (`custom_providers` block pointing at halogen with server-side `HALOGEN_*` defaults, per the halogen README's harness guidance); limitations section from the spec (no call-purpose thinking override for compaction, env read at process start, static vision flag, user-set `max_tokens` above the clamp relies on context_overflow recovery).

- [ ] **Step 4: Commit** `docs: README and Docker e2e verification for halogen provider`

---

## Self-review notes

- Spec coverage: wire mapping (T1), errors (T2), profile/catalog/vision/aux/client (T3), live smoke (T4), e2e + docs + fallback + limitations (T5). Retry/stream-idle/vision-probe are spec'd out of scope — no tasks needed.
- Review Focus 1–5 each pinned in T1/T2/T3 as noted inline.
- Type consistency: `resolve_thinking(reasoning_config, *, default_effort, preserve_thinking) -> dict` and `clamp_output_budget(max_tokens, context_window) -> int | None` used identically in T1/T3/T4; `classify_halogen_error` named identically in T2/T3.
