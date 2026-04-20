.DEFAULT_GOAL := help
SHELL := /bin/bash

# Makefile is a thin wrapper around scripts/*.sh so either interface works.
# Prefer the shell scripts directly if you're automating things — they
# print usage on -h and are easier to grep.

# =============================================================================
# Help
# =============================================================================
.PHONY: help
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Setup
# =============================================================================
.PHONY: setup
setup:  ## First-time setup (env + deps + models)
	bash scripts/setup.sh

.PHONY: setup-no-models
setup-no-models:  ## Setup without downloading models
	bash scripts/setup.sh --no-models

.PHONY: env
env:  ## Create .env from template if missing
	@test -f .env || (cp .env.example .env && echo "Created .env — edit it.")

.PHONY: install
install:  ## Install backend deps locally with uv
	uv sync --extra dev

.PHONY: download-models
download-models:  ## Pre-pull HF model weights into the shared cache volume
	bash scripts/download_models.sh

# =============================================================================
# Compose lifecycle
# =============================================================================
.PHONY: start
start:  ## Bring up all services (vLLM Qwen + Whisper + backend)
	bash scripts/start.sh

.PHONY: start-dev
start-dev:  ## Start with dev overrides (hot reload, DEBUG logs)
	bash scripts/start.sh --dev

.PHONY: start-follow
start-follow:  ## Start and tail logs
	bash scripts/start.sh --follow

.PHONY: stop
stop:  ## Stop and remove containers
	bash scripts/stop.sh

.PHONY: stop-all
stop-all:  ## Stop, remove containers AND volumes
	bash scripts/stop.sh --all

.PHONY: restart
restart: stop start  ## Restart everything

.PHONY: status
status:  ## Show stack status (containers + /v1/models + /readyz + GPU)
	bash scripts/status.sh

.PHONY: ps
ps:  ## Raw `docker compose ps` (use `make status` for health + endpoints)
	docker compose ps

.PHONY: logs
logs:  ## Tail all logs
	bash scripts/logs.sh

.PHONY: logs-backend
logs-backend:  ## Tail backend logs
	bash scripts/logs.sh backend

.PHONY: logs-qwen
logs-qwen:  ## Tail Qwen vLLM logs
	bash scripts/logs.sh qwen

.PHONY: logs-whisper
logs-whisper:  ## Tail Whisper vLLM logs
	bash scripts/logs.sh whisper

.PHONY: shell-backend
shell-backend:  ## Shell into backend container
	docker compose exec backend /bin/bash

# =============================================================================
# Testing
# =============================================================================
.PHONY: test
test:  ## Run ruff + mypy + pytest
	bash scripts/test.sh

.PHONY: test-unit
test-unit:  ## Run pytest only
	bash scripts/test.sh unit

.PHONY: test-cov
test-cov:  ## Run pytest with coverage
	bash scripts/test.sh cov

.PHONY: smoke
smoke:  ## End-to-end smoke test against running services
	bash scripts/smoke_test.sh

.PHONY: bench
bench:  ## WER benchmark on data/samples audio pairs
	bash scripts/bench.sh

# =============================================================================
# Quality
# =============================================================================
.PHONY: lint
lint:  ## Run ruff + mypy
	bash scripts/test.sh lint
	bash scripts/test.sh type

.PHONY: fmt
fmt:  ## Auto-format with ruff
	bash scripts/test.sh fmt

.PHONY: check
check: lint test-unit  ## Lint + test (run before committing)

# =============================================================================
# Clean
# =============================================================================
.PHONY: clean
clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean stop-all  ## clean + remove containers and volumes
