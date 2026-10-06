# Halogen provider plugin for Hermes Agent

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) model-provider
plugin that routes Hermes inference through a
[halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)
instance — the OpenAI-compatible server that runs Qwen3.8-Flash-Next at native
speed on AMD Strix Halo.

Out of the box Hermes treats a custom OpenAI-compatible endpoint as a plain
chat-completions target: no thinking control, no halogen-aware error handling.
This plugin registers a `halogen` provider (alias `halo`) that adds what the
generic route lacks:

- **Thinking control on the wire** — Hermes' reasoning config maps onto
  halogen's `enable_thinking` / `reasoning_effort` fields (sent via
  `extra_body`), with `preserve_thinking` passback when enabled.
- **Output-budget clamp** — `max_tokens` is capped at
  `min(requested, max(1024, 25% × context))` so `max_tokens + prompt ≤ context`
  holds (halogen admits a request only when that is true).
- **halogen error taxonomy** — 401/403 → auth, 429 → rate limit, 5xx →
  server/overloaded, and halogen's token-budget 400 wording ("does not fit",
  "leaving room for") → `context_overflow` so Hermes' compaction can recover
  instead of aborting the turn.
- **Catalog** — model `halogen-qwen3.8-flash-next`, context 262,144, tools and
  reasoning declared.

The repo root **is** the plugin directory (`plugin.yaml` + `__init__.py`), so
any of the three install paths below works directly. No Python dependencies
beyond the standard library.

## Install

Pick one.

**1. Drop-in copy** (local Hermes install):

```bash
git clone <this-repo> && cd halogen-hermes
mkdir -p ~/.hermes/plugins/model-providers/halogen
tar cf - --exclude=./tests --exclude=./scripts --exclude=./docs \
    --exclude=./.git --exclude=./conftest.py --exclude='__pycache__' . \
  | (cd ~/.hermes/plugins/model-providers/halogen && tar xf -)
```

**2. Catalog install** (once the repo is published):

```bash
hermes plugins install <owner>/halogen-hermes
```

**3. Docker bind-mount** (throwaway container, as `scripts/docker_verify.sh`
does it): stage the plugin at `$HERMES_HOME/plugins/model-providers/halogen/`
inside the mounted data dir (`/opt/data` in the official image) — see
[Verify](#verify) for a working command.

Then select the provider (any Hermes config surface works — `hermes model`,
`--provider halogen -m halogen-qwen3.8-flash-next`, or `config.yaml`):

```yaml
# ~/.hermes/config.yaml
model:
  provider: halogen
  name: halogen-qwen3.8-flash-next   # `name` is an alias of `default`
```

## Configuration

All settings are environment variables, read once when Hermes constructs the
provider profile at process start:

| Variable | Default | Meaning |
| --- | --- | --- |
| `HALOGEN_BASE_URL` | `http://127.0.0.1:8731/v1` | endpoint base |
| `HALOGEN_API_KEY` | `no-key-required` placeholder | halogen itself has no auth, but Hermes' credential gate refuses an empty key for an api-key provider — set any non-empty value; it is sent as a bearer token halogen ignores. Point it at a real token only behind a bearer-authenticated proxy. |
| `HALOGEN_REASONING_EFFORT` | `medium` | default effort when Hermes sends none (`minimal`/`low`/`medium`/`high`/`xhigh`, or `off`) |
| `HALOGEN_PRESERVE_THINKING` | unset | `1` adds `preserve_thinking: true` (reasoning passback, cache reuse) |
| `HALOGEN_MAX_TOKENS` | `8192` | provider-level output cap (server cap is 65536) |
| `HALOGEN_CTX` | `262144` | declared context window (YaRN deployments raise it) |
| `HALOGEN_VISION` | unset | `1` declares `supports_vision: true` for the catalog model |

Malformed values fall back to defaults — a bad environment never crashes
Hermes startup. Changing any of these means restarting Hermes.

## Verify

Two scripts, no Hermes install needed for the first:

```bash
# 1. wire smoke — proves the halogen endpoint speaks what the plugin sends
HALOGEN_BASE_URL=http://192.168.31.7:8731/v1 python3 scripts/wire_smoke.py
# PASS models / chat-thinking-on / chat-thinking-off / tool-round-trip

# 2. container e2e — real Hermes image, plugin bind-mounted, one real turn
HALOGEN_BASE_URL=http://192.168.31.7:8731/v1 bash scripts/docker_verify.sh
# PASS: halogen provider discovered
# PASS: token HALOGEN_E2E_OK found (hermes chat -q)
```

`docker_verify.sh` stages a throwaway `mktemp` `HERMES_HOME` (removed on
exit), runs `hermes plugins list` (halogen must appear), then a one-shot turn
`"Reply with exactly: HALOGEN_E2E_OK"` and requires the token in the output.
That list shows the row as `not enabled  user  0.1.0  halogen-provider` even
when the plugin is installed and working: for a drop-in model-provider plugin
that status label is expected, not an install failure — the provider is active
once selected via config/env, which is why the script only requires the name to
appear. Notes from the image it was verified against
(`nousresearch/hermes-agent:latest`):

- The image's one-shot CLI is `hermes chat -q "<prompt>"` or `hermes -z
  "<prompt>"` (prints only the final response); a top-level `hermes -q` is
  not accepted, so the script tries the forms in order.
- On SELinux-enforcing hosts the bind mount needs the `:z` relabel suffix
  (the script adds it automatically).
- The container's bootstrap chowns the data dir to its internal `hermes`
  user; on rootless engines the script chowns it back before cleanup.

## Zero-code fallback (no plugin)

If you want halogen without installing anything, Hermes' built-in custom
provider route works — with one loss: no thinking control, so the **server's
defaults are what you run at** (per the halogen README's harness guidance,
agent harnesses send no thinking fields to a custom endpoint unless told to).

```yaml
# ~/.hermes/config.yaml
model:
  provider: halogen
  default: halogen-qwen3.8-flash-next

custom_providers:
  - name: halogen
    base_url: http://192.168.31.7:8731/v1
    model: halogen-qwen3.8-flash-next
    context_length: 262144
    api_key: "no-key-required"   # halogen has no auth; Hermes needs a non-empty key
    extra_body:                  # optional: pin thinking statically for every call
      enable_thinking: true
      reasoning_effort: medium
```

For server-side defaults instead of (or on top of) `extra_body`, run
halogen-flash-server with its own `HALOGEN_*` variables — they fill fields a
request omits: `HALOGEN_REASONING_EFFORT` (effort when none is named),
`HALOGEN_ENABLE_THINKING=0` (thinking off unless a request turns it on),
`HALOGEN_MAX_THINKING_TOKENS` (budget for requests that send none). Hermes'
`extra_body` above wins per-request where both apply.

## Limitations

- **No call-purpose thinking override.** The reference DSH adapter forces
  thinking off for cheap calls like session titles; Hermes exposes no
  call-purpose seam to provider profiles, so compaction summaries and
  auxiliary calls run at the configured effort. Mitigate server-side
  (`HALOGEN_MAX_THINKING_TOKENS`) or accept the cost.
- **Env config is read at process start.** No live reload; restart Hermes
  after changing `HALOGEN_*` variables.
- **`supports_vision` is a static declaration** gated on `HALOGEN_VISION=1`,
  not probed from the server's `/health`. Set it only when the deployment
  actually runs the vision tower (`HALOGEN_VISION_TOWER=1`), or image-bearing
  requests will 400.
- **A user-set `max_tokens` above the clamp is not silently rewritten.** The
  plugin clamps its own default budget; an explicit higher cap from the user
  relies on halogen's budget-400 → `context_overflow` classification to make
  Hermes compact and retry rather than fail the turn.
