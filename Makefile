.PHONY: help install install-dev test lint format type-check check clean ci

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	uv sync --frozen

install-dev: ## Install with dev dependencies
	uv sync --frozen --all-extras

test: ## Run tests
	uv run pytest -v

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html -v

test-fast: ## Run tests in parallel
	uv run pytest -n auto -v

lint: ## Run linter
	uv run ruff check .

lint-fix: ## Run linter and fix issues
	uv run ruff check --fix .

format: ## Format code
	uv run ruff format .

format-check: ## Check code formatting
	uv run ruff format --check .

type-check: ## Run type checker
	uv run mypy src tests

check: lint format-check type-check test ## Run all checks (lint, format, type, test)

clean: ## Clean cache and build files
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

ci: ## Run CI checks locally (requires act)
	./scripts/run_ci_locally.sh

lock: ## Update lockfile
	uv lock --upgrade

sync: ## Sync environment with lockfile
	uv sync --frozen --all-extras

update: lock sync ## Update dependencies and sync environment
