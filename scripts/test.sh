#!/usr/bin/env bash
# Run backend unit tests + lint + type check. Doesn't require services running.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mode="${1:-all}"

case "${mode}" in
    unit|u)   uv run pytest backend/tests -v ;;
    lint|l)   uv run ruff check backend/src backend/tests ;;
    type|t)   uv run mypy ;;
    fmt|f)    uv run ruff format backend/src backend/tests
              uv run ruff check --fix backend/src backend/tests ;;
    cov)      uv run pytest --cov --cov-report=term-missing ;;
    all|*)
        echo "== ruff =="
        uv run ruff check backend/src backend/tests
        echo "== mypy =="
        uv run mypy
        echo "== pytest =="
        uv run pytest backend/tests -v
        ;;
esac
