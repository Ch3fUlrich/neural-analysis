#!/usr/bin/env bash
set -euo pipefail

# setup_env.sh - idempotent script to bootstrap dev environment for this project
# Usage:
#   UV_INSTALL_METHOD=installer ./scripts/setup_env.sh    # default installer
#   UV_INSTALL_METHOD=pipx ./scripts/setup_env.sh         # use pipx
#   UV_INSTALL_METHOD=pip ./scripts/setup_env.sh         # pip user install
#   RUN_TESTS=1 ./scripts/setup_env.sh                    # run linters/tests after sync
#   UV_FORCE_LOCK=1 ./scripts/setup_env.sh                # force lock refresh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

UV_INSTALL_METHOD=${UV_INSTALL_METHOD:-installer}

echo "Project root: $PROJECT_ROOT"
echo "UV install method: $UV_INSTALL_METHOD"

if command -v apt >/dev/null 2>&1; then
  echo "Installing system build dependencies (may ask for sudo)..."
  sudo apt update
  sudo apt install -y build-essential python3-dev libssl-dev libffi-dev curl ca-certificates || true
else
  echo "No apt found; ensure build toolchain and python dev headers are installed as needed."
fi

if command -v uv >/dev/null 2>&1; then
  echo "uv already available: $(command -v uv)"
else
  echo "uv not found; installing via: $UV_INSTALL_METHOD"
  case "$UV_INSTALL_METHOD" in
    installer)
      echo "Using standalone installer (curl | sh)..."
      curl -sSf https://install.astral.sh | sh
      ;;
    pipx)
      echo "Installing via pipx..."
      python3 -m pip install --user pipx
      python3 -m pipx ensurepath || true
      # reload path for current shell
      export PATH="$HOME/.local/bin:$PATH"
      pipx install uv
      ;;
    pip)
      echo "Installing via pip (user)..."
      python3 -m pip install --user --upgrade pip
      python3 -m pip install --user uv
      export PATH="$HOME/.local/bin:$PATH"
      ;;
    *)
      echo "Unknown UV_INSTALL_METHOD: $UV_INSTALL_METHOD" >&2
      exit 1
      ;;
  esac
fi

# Ensure local user bin is on PATH for the rest of the script
export PATH="$HOME/.local/bin:$PATH"

echo "uv version: $(uv --version || true)"

# Decide whether to install development packages (early so lock generation can include them)
if [ -z "${INSTALL_DEV:-}" ]; then
  # interactive prompt
  read -r -p "Install development packages (dev group)? [y/N]: " _ans
  case "$_ans" in
    [Yy]* ) INSTALL_DEV=1 ;;
    * ) INSTALL_DEV=0 ;;
  esac
fi

if [ ! -f uv.lock ] || [ "${UV_FORCE_LOCK:-0}" = "1" ]; then
  echo "Generating lockfile (uv.lock)..."
  if [ "${INSTALL_DEV}" = "1" ]; then
    echo "Including development group in lockfile (uv.lock will pick up pyproject extras)..."
    # `uv lock` reads pyproject.toml and will include optional extras defined there.
    uv lock
  else
    uv lock
  fi
else
  # uv.lock exists. If the user requested dev packages, ensure the lockfile includes them; otherwise skip.
  if [ "${INSTALL_DEV}" = "1" ]; then
    echo "uv.lock exists — checking whether dev packages are present in lockfile..."
    missing_dev=0
    for pkg in ruff mypy pytest pytest-cov pre-commit flake8 pydocstyle pytest-xdist coverage; do
      if ! grep -F "name = \"$pkg\"" uv.lock >/dev/null 2>&1; then
        echo "  dev package missing from lockfile: $pkg"
        missing_dev=1
      fi
    done
    if [ "$missing_dev" = "1" ]; then
      echo "Some dev packages are not present in uv.lock — regenerating lockfile to include dev group..."
      uv lock
    else
      echo "uv.lock already contains dev packages; skipping lock regeneration."
    fi
  else
    echo "uv.lock already exists; skip lock. Set UV_FORCE_LOCK=1 to force refresh."
  fi
fi

echo "Syncing environment (installing dependencies)..."
if [ "${INSTALL_DEV}" = "1" ] || [ "${INSTALL_DEV}" = "yes" ]; then
  echo "Including development dependency group..."
  uv sync --all-groups
else
  echo "Skipping development dependency group (dev)."
  uv sync --no-dev
fi

if [ "${INSTALL_DEV}" = "1" ]; then
  echo "Installing pre-commit hooks via uv-run (if available)..."
  # Prefer running pre-commit inside the uv-managed environment so the correct
  # package is used. This ensures hooks are installed into the repo and usable
  # locally.
  # Only attempt to install hooks if git is present and we're inside a repo.
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Running: uv run -- pre-commit install"
    uv run -- pre-commit install || true
  else
    echo "Git not found or not inside a git repository; skipping pre-commit install."
    echo "If you want hooks installed, run this script from a checked-out git clone."
  fi
else
  echo "Dev group not installed; skipping pre-commit installation."
fi

if [ "${RUN_TESTS:-0}" = "1" ]; then
  echo "Running linters and tests (RUN_TESTS=1)..."
  uv run ruff . || true
  uv run mypy src tests || true
  uv run pytest -q || true
fi

echo "Bootstrapping complete. Activate the project venv (if any) and start developing." 
