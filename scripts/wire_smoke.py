#!/usr/bin/env python3
"""Live wire smoke test for a halogen-flash-server endpoint.

Proves the wire contract the Hermes halogen provider relies on against a
real halogen instance: model availability, a chat completion with thinking
on, one with thinking off, and a tool-call round-trip. Doubles as a
user-facing diagnostic for a new deployment.

Stdlib only (urllib). Run from anywhere:

    HALOGEN_BASE_URL=http://192.168.31.7:8731/v1 python3 scripts/wire_smoke.py

Exit code is 0 only when all four checks pass.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Make `import wire` work from any CWD by putting the repo root (the parent
# of this script's directory) on sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wire import resolve_thinking  # noqa: E402

BASE_URL = os.environ.get("HALOGEN_BASE_URL", "http://127.0.0.1:8731/v1").rstrip("/")
MODEL = "halogen-qwen3.8-flash-next"
TIMEOUT_S = 120

WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _snippet(body, limit=300):
    text = json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body
    return text if len(text) <= limit else text[:limit] + "..."


def _request(path, payload=None):
    """GET (payload None) or POST JSON to {BASE_URL}{path}.

    Returns (status, parsed_body). Non-2xx responses come back through
    HTTPError with their parsed body so failures can quote the server's
    error message; transport errors raise to the caller.
    """
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def _chat(messages, extra=None, max_tokens=64):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    return _request("/chat/completions", payload)


def _first_message(body):
    choices = body.get("choices") if isinstance(body, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return None


def check_models():
    status, body = _request("/models")
    if status != 200:
        return False, f"GET /models -> HTTP {status}: {_snippet(body)}"
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return False, f"GET /models -> 200 but no data list: {_snippet(body)}"
    ids = [m.get("id") for m in data if isinstance(m, dict)]
    if MODEL in ids:
        return True, f"{MODEL} listed in /models"
    return False, f"{MODEL} not in model list {ids}"


def check_chat_thinking_on():
    thinking = resolve_thinking({"enabled": True, "effort": "minimal"})
    status, body = _chat(
        [{"role": "user", "content": "Reply with exactly: HALOGEN_SMOKE_OK"}],
        extra=thinking,
        max_tokens=256,
    )
    if status != 200:
        return False, f"chat thinking-on -> HTTP {status}: {_snippet(body)}"
    message = _first_message(body)
    content = (message or {}).get("content") or ""
    if not content.strip():
        return False, f"chat thinking-on -> 200 but empty content: {_snippet(body)}"
    return True, f"chat thinking-on {thinking} -> 200, content {content.strip()[:60]!r}"


def check_chat_thinking_off():
    thinking = resolve_thinking({"enabled": False})
    status, body = _chat(
        [{"role": "user", "content": "Reply with exactly: HALOGEN_SMOKE_OK"}],
        extra=thinking,
    )
    if status != 200:
        return False, f"chat thinking-off -> HTTP {status}: {_snippet(body)}"
    return True, f"chat thinking-off {thinking} -> 200"


def check_tool_round_trip():
    status, body = _chat(
        [{"role": "user", "content": "What's the weather in Lisbon?"}],
        extra={"tools": WEATHER_TOOL},
        max_tokens=256,
    )
    if status != 200:
        return False, f"tool round-trip -> HTTP {status}: {_snippet(body)}"
    message = _first_message(body)
    tool_calls = (message or {}).get("tool_calls") or []
    if not tool_calls:
        content = (message or {}).get("content") or ""
        return False, f"tool round-trip -> 200 but no tool_calls; content {content[:80]!r}"
    names = [(tc.get("function") or {}).get("name") for tc in tool_calls if isinstance(tc, dict)]
    return True, f"tool round-trip -> 200, tool_calls {names}"


def main():
    checks = [
        ("models", check_models),
        ("chat-thinking-on", check_chat_thinking_on),
        ("chat-thinking-off", check_chat_thinking_off),
        ("tool-round-trip", check_tool_round_trip),
    ]
    print(f"halogen wire smoke against {BASE_URL} (model {MODEL})", file=sys.stderr)
    failures = 0
    for label, check in checks:
        try:
            ok, detail = check()
        except Exception as exc:  # transport, timeout, undecodable body
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}")
        failures += 0 if ok else 1
    if failures:
        print(f"{failures} of {len(checks)} checks failed", file=sys.stderr)
        return 1
    print(f"all {len(checks)} checks passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
