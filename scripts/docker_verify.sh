#!/usr/bin/env bash
# End-to-end verification of the halogen model-provider plugin inside the
# official Hermes container: drop the plugin into a throwaway $HERMES_HOME,
# point Hermes at a live halogen endpoint, and prove a real one-shot turn
# routes through the plugin.
#
# Usage:
#   HALOGEN_BASE_URL=http://192.168.31.7:8731/v1 bash scripts/docker_verify.sh
#
# Env:
#   HALOGEN_BASE_URL   required — a halogen endpoint reachable *from the
#                      container* (127.0.0.1 inside the container is not this host)
#   HALOGEN_API_KEY    default "no-key-required" — halogen itself has no auth,
#                      but Hermes' api-key credential gate refuses an empty key
#                      for an auth_type=api_key provider, so any non-empty value
#                      passes (the profile sends it as a bearer halogen ignores)
#   IMAGE              image to run (default nousresearch/hermes-agent:latest)
#   DOCKER             container CLI (default docker; falls back to podman)
#
# Nothing persists on the host: the data dir is a mktemp tree removed on exit.
set -euo pipefail

HALOGEN_BASE_URL=${HALOGEN_BASE_URL:?"set HALOGEN_BASE_URL to a halogen /v1 endpoint reachable from the container, e.g. http://192.168.31.7:8731/v1"}
HALOGEN_API_KEY=${HALOGEN_API_KEY:-no-key-required}
IMAGE=${IMAGE:-nousresearch/hermes-agent:latest}
MODEL=${MODEL:-halogen-qwen3.8-flash-next}
TOKEN=HALOGEN_E2E_OK

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# --- container CLI: docker, falling back to podman when docker errors ---
DOCKER=${DOCKER:-docker}
if ! "$DOCKER" info >/dev/null 2>&1; then
    if command -v podman >/dev/null 2>&1; then
        echo "note: '$DOCKER' unusable, falling back to podman" >&2
        DOCKER=podman
    else
        echo "error: '$DOCKER' is not usable and podman is not installed" >&2
        exit 1
    fi
fi

# SELinux-enforcing hosts (Fedora/RHEL) refuse container reads of unlabelled
# bind mounts; podman/docker accept the :z relabel suffix.
MOUNT_SUFFIX=""
if command -v getenforce >/dev/null 2>&1 && [ "$(getenforce 2>/dev/null || true)" = "Enforcing" ]; then
    MOUNT_SUFFIX=":z"
fi

DATA=$(mktemp -d)
cleanup() {
    # The container's stage2 hook chowns $HERMES_HOME to its hermes user, which
    # on rootless engines lands as an unmappable host uid — hand the tree back
    # to this host user (container root under rootless) before retrying.
    rm -rf "$DATA" 2>/dev/null || true
    if [ -d "$DATA" ]; then
        "$DOCKER" run --rm --entrypoint sh \
            -v "$DATA:/cleanup$MOUNT_SUFFIX" "$IMAGE" \
            -c 'chown -R 0:0 /cleanup' >/dev/null 2>&1 || true
        rm -rf "$DATA" 2>/dev/null || true
    fi
    [ -d "$DATA" ] && echo "warning: could not remove $DATA" >&2
    return 0
}
trap cleanup EXIT

# --- stage the plugin into a throwaway HERMES_HOME ---
# The repo root IS the plugin directory; ship everything except dev-only trees.
PLUGIN_DIR="$DATA/plugins/model-providers/halogen"
mkdir -p "$PLUGIN_DIR"
(
    cd "$REPO_ROOT"
    tar cf - \
        --exclude=./tests \
        --exclude=./scripts \
        --exclude=./docs \
        --exclude=./.git \
        --exclude=./.venv \
        --exclude=./.superpowers \
        --exclude=./conftest.py \
        --exclude=./__pycache__ \
        --exclude='__pycache__' \
        .
) | (cd "$PLUGIN_DIR" && tar xf -)
# Repo files may be owner-only (600); the container's hermes user must read the
# plugin and write into $HERMES_HOME (auth.json, .env seeding).
chmod -R a+rwX "$DATA"

# Hermes accepts model.name as an alias of model.default (config.py normalizes
# default > model > name), so this shape is valid.
cat > "$DATA/config.yaml" <<YAML
model:
  provider: halogen
  name: $MODEL
YAML

echo "staged plugin files:"
ls "$PLUGIN_DIR"

run_hermes() {
    "$DOCKER" run --rm \
        -v "$DATA:/opt/data$MOUNT_SUFFIX" \
        -e HALOGEN_BASE_URL="$HALOGEN_BASE_URL" \
        -e HALOGEN_API_KEY="$HALOGEN_API_KEY" \
        "$IMAGE" hermes "$@"
}

# halogen counts as discovered only when it appears AND no load failure names it
# (a "Failed to load user provider plugin halogen" line would match a bare grep).
halogen_discovered() {
    printf '%s\n' "$1" | grep -qi halogen &&
        ! printf '%s\n' "$1" | grep -q "Failed to load user provider plugin halogen"
}

# --- check 1: the plugin is discovered ---
echo "== hermes plugins list =="
list_out=$(run_hermes plugins list --plain --no-bundled 2>&1) || list_out_rc=$?
if [ "${list_out_rc:-0}" -ne 0 ] || ! halogen_discovered "$list_out"; then
    printf '%s\n' "$list_out"
    echo "note: 'hermes plugins list' unusable; falling back to hermes doctor" >&2
    if ! doctor_out=$(run_hermes doctor 2>&1); then
        printf '%s\n' "$doctor_out" >&2
        echo "FAIL: both 'plugins list' and 'doctor' failed"
        exit 1
    fi
    printf '%s\n' "$doctor_out"
    halogen_discovered "$doctor_out" || { echo "FAIL: halogen not discovered"; exit 1; }
else
    printf '%s\n' "$list_out"
fi
echo "PASS: halogen provider discovered"

# --- check 2: a real one-shot turn routes through the plugin ---
# The brief's `hermes -q "<prompt>"` is the shape verified from Hermes source;
# the current image exposes one-shot as `hermes chat -q` / `hermes -z` instead,
# so try the documented shapes in order and require the token from one.
echo "== one-shot turn via halogen =="
for attempt in "-q" "chat -q" "-z"; do
    # shellcheck disable=SC2086  # intentional word splitting of the attempt form
    if query_out=$(run_hermes $attempt "Reply with exactly: $TOKEN" 2>&1) &&
        printf '%s\n' "$query_out" | grep -q "$TOKEN"; then
        printf '%s\n' "$query_out"
        echo "PASS: token $TOKEN found (hermes $attempt)"
        echo "docker e2e OK ($IMAGE, $HALOGEN_BASE_URL)"
        exit 0
    fi
    printf 'note: `hermes %s` did not yield the token, trying next form\n' "$attempt" >&2
done
printf '%s\n' "$query_out" >&2
echo "FAIL: token $TOKEN not found in any one-shot form"
exit 1
