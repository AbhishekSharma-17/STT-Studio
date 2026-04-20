# syntax=docker/dockerfile:1.7
# Multi-stage backend image with uv for fast, reproducible installs.
ARG PYTHON_VERSION=3.12

# ---- base: shared system deps --------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv provides deterministic installs and is ~10x faster than pip here
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# ---- deps: install dependencies into /opt/venv --------------------------------
FROM base AS deps
COPY pyproject.toml /app/
# Install into a venv we'll copy forward; this layer only rebuilds when
# pyproject.toml changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --python=/opt/venv/bin/python -r pyproject.toml

# ---- dev: hot-reload with source mounted at runtime ---------------------------
FROM deps AS dev
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src"
RUN VIRTUAL_ENV=/opt/venv uv pip install --python=/opt/venv/bin/python \
    --extra-index-url https://pypi.org/simple -r pyproject.toml --extra dev
# Source is bind-mounted by compose.dev.yml
CMD ["uvicorn", "stt_backend.main:app", "--host=0.0.0.0", "--port=3000", "--reload"]

# ---- runtime: copy source + frontend, run as non-root --------------------------
FROM base AS runtime
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

COPY --from=deps /opt/venv /opt/venv
COPY backend/src /app/src
COPY frontend /app/frontend

RUN groupadd --system app && useradd --system --gid app --create-home app && \
    chown -R app:app /app
USER app

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3000/healthz || exit 1

CMD ["uvicorn", "stt_backend.main:app", "--host=0.0.0.0", "--port=3000"]
