#!/usr/bin/env bash
# Report the live status of the STT stack.
#
# Checks, in order:
#   1. Running containers (docker compose ps)
#   2. vLLM /v1/models for Qwen and Whisper
#   3. Backend /healthz and /readyz
#   4. GPU memory usage (nvidia-smi) if available
#
# Exit codes:
#   0  everything healthy and ready
#   1  at least one component is not ready
#
# Usage:
#   bash scripts/status.sh          # human-readable
#   bash scripts/status.sh --quiet  # only exit code + one-line summary
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Load .env if present (for BACKEND_PORT / QWEN_PORT / WHISPER_PORT)
if [[ -f .env ]]; then
    # shellcheck disable=SC1091
    set -a && source .env && set +a
fi

BACKEND_URL="http://localhost:${BACKEND_PORT:-3000}"
QWEN_URL="http://localhost:${QWEN_PORT:-8000}"
WHISPER_URL="http://localhost:${WHISPER_PORT:-8001}"

QUIET=0
for arg in "$@"; do
    case "${arg}" in
        -q|--quiet) QUIET=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    esac
done

# --- Pretty-printing helpers -------------------------------------------------
if [[ -t 1 ]] && (( QUIET == 0 )); then
    GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'
    DIM=$'\033[2m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    GREEN=""; RED=""; YELLOW=""; DIM=""; BOLD=""; RESET=""
fi

ok()   { (( QUIET )) || echo -e "  ${GREEN}✓${RESET} $*"; }
warn() { (( QUIET )) || echo -e "  ${YELLOW}!${RESET} $*"; }
bad()  { (( QUIET )) || echo -e "  ${RED}✗${RESET} $*"; }
hdr()  { (( QUIET )) || echo -e "\n${BOLD}$*${RESET}"; }

overall_ok=1
mark_bad() { overall_ok=0; }

# --- 1. docker compose ps ----------------------------------------------------
hdr "Containers"
if ! docker compose ps --format '{{.Service}} {{.Status}}' 2>/dev/null | \
     awk 'NF{ print }' > /tmp/stt-ps.$$ 2>/dev/null; then
    bad "docker compose not reachable"
    mark_bad
else
    if [[ ! -s /tmp/stt-ps.$$ ]]; then
        bad "no services running — try: bash scripts/start.sh"
        mark_bad
    else
        while read -r svc status; do
            if [[ "${status}" == *"healthy"* ]]; then
                ok "${svc} — ${status}"
            elif [[ "${status}" == *"Up"* ]]; then
                warn "${svc} — ${status}  (no healthcheck yet)"
            else
                bad "${svc} — ${status}"
                mark_bad
            fi
        done < /tmp/stt-ps.$$
    fi
fi
rm -f /tmp/stt-ps.$$

# --- 2. vLLM endpoints -------------------------------------------------------
probe_vllm() {
    local name="$1" url="$2"
    local resp
    if resp=$(curl -fsS -m 3 "${url}/v1/models" 2>/dev/null); then
        local model
        model=$(printf '%s' "${resp}" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo "?")
        ok "${name}: serving ${DIM}${model}${RESET} at ${url}"
    else
        bad "${name}: unreachable at ${url}/v1/models"
        mark_bad
    fi
}

hdr "vLLM servers"
probe_vllm "qwen   " "${QWEN_URL}"
probe_vllm "whisper" "${WHISPER_URL}"

# --- 3. Backend --------------------------------------------------------------
hdr "Backend"
if curl -fsS -m 3 "${BACKEND_URL}/healthz" >/dev/null 2>&1; then
    ok "healthz: 200"
else
    bad "healthz: unreachable at ${BACKEND_URL}/healthz"
    mark_bad
fi

if body=$(curl -fsS -m 5 "${BACKEND_URL}/readyz" 2>/dev/null); then
    ready=$(printf '%s' "${body}" | python3 -c 'import sys,json; print("yes" if json.load(sys.stdin)["ready"] else "no")' 2>/dev/null || echo "?")
    if [[ "${ready}" == "yes" ]]; then
        ok "readyz: ready"
    else
        bad "readyz: not ready — $(printf '%s' "${body}" | head -c 200)"
        mark_bad
    fi
else
    http=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "${BACKEND_URL}/readyz" 2>/dev/null || echo "??")
    bad "readyz: HTTP ${http} at ${BACKEND_URL}/readyz"
    mark_bad
fi

# --- 4. GPU memory (optional) ------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    hdr "GPU"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | \
    while IFS=, read -r idx name mem_used mem_total util; do
        (( QUIET )) || echo "  [${idx# }] ${name# } · VRAM${mem_used} /${mem_total} · util${util}"
    done
fi

# --- Summary -----------------------------------------------------------------
echo
if (( overall_ok )); then
    echo -e "${GREEN}${BOLD}READY${RESET} — stack is healthy."
    exit 0
else
    echo -e "${RED}${BOLD}NOT READY${RESET} — see failures above. Logs: bash scripts/logs.sh"
    exit 1
fi
