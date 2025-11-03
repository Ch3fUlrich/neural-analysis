#!/usr/bin/env bash
set -euo pipefail

# setup_env.sh - idempotent script to bootstrap dev environment for this project
# Usage:
#   UV_INSTALL_METHOD=installer ./scripts/setup_env.sh    # default installer
#   UV_INSTALL_METHOD=pipx ./scripts/setup_env.sh         # use pipx
#   UV_INSTALL_METHOD=pip ./scripts/setup_env.sh         # pip user install
#   INSTALL_DEV=1 ./scripts/setup_env.sh                  # install all dev tools (packages, act, docker)
#   INSTALL_DEV=0 ./scripts/setup_env.sh                  # minimal installation (no dev tools)
#   RUN_TESTS=1 ./scripts/setup_env.sh                    # run linters/tests after sync
#   UV_FORCE_LOCK=1 ./scripts/setup_env.sh                # force lock refresh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Interactive mode detection
INTERACTIVE_MODE=0
if [ -z "${INSTALL_DEV:-}" ] && [ -z "${UV_INSTALL_METHOD:-}" ]; then
  INTERACTIVE_MODE=1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Neural Analysis - Environment Setup Script            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Interactive configuration menu
if [ "$INTERACTIVE_MODE" = "1" ]; then
  echo "==> Interactive Setup Mode"
  echo ""
  echo "This script will help you set up your development environment."
  echo "You can also run this script non-interactively by setting environment variables."
  echo "See the script header for usage examples."
  echo ""
  
  # Ask about uv installation method
  if ! command -v uv >/dev/null 2>&1; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1. UV Package Manager Installation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "uv is not installed. Choose installation method:"
    echo "  1) installer (recommended) - standalone installer"
    echo "  2) pipx - install via pipx"
    echo "  3) pip - install via pip"
    read -r -p "Select method [1-3] (default: 1): " uv_method_choice
    case "$uv_method_choice" in
      2) UV_INSTALL_METHOD="pipx" ;;
      3) UV_INSTALL_METHOD="pip" ;;
      *) UV_INSTALL_METHOD="installer" ;;
    esac
  else
    UV_INSTALL_METHOD="skip"
    echo "✓ uv already installed: $(command -v uv)"
  fi
  
  # Ask about development environment
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "2. Development Environment"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Install full development environment?"
  echo "This includes:"
  echo "  - Python packages: pytest, mypy, ruff, pre-commit, etc."
  echo "  - act: Run GitHub Actions CI locally"
  echo "  - Docker: Required by act (if not already installed)"
  read -r -p "Install development environment? [Y/n]: " dev_choice
  case "$dev_choice" in
    [Nn]* ) INSTALL_DEV=0 ;;
    * ) INSTALL_DEV=1 ;;
  esac
  
  # Ask about running tests after setup
  if [ "$INSTALL_DEV" = "1" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "3. Post-Setup Validation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Run linters and tests after setup to verify installation?"
    read -r -p "Run validation checks? [y/N]: " run_tests_choice
    case "$run_tests_choice" in
      [Yy]* ) RUN_TESTS=1 ;;
      * ) RUN_TESTS=0 ;;
    esac
  fi
  
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Configuration Summary:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  UV install method:         $UV_INSTALL_METHOD"
  echo "  Install dev environment:   $INSTALL_DEV"
  echo "  Run validation:            ${RUN_TESTS:-0}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  read -r -p "Proceed with installation? [Y/n]: " proceed_choice
  case "$proceed_choice" in
    [Nn]* )
      echo "Setup cancelled by user."
      exit 0
      ;;
  esac
  echo ""
else
  # Non-interactive mode - use environment variables
  UV_INSTALL_METHOD=${UV_INSTALL_METHOD:-installer}
  echo "==> Non-Interactive Mode"
  echo "UV install method: $UV_INSTALL_METHOD"
fi

echo "==> Installing System Dependencies"
if command -v apt >/dev/null 2>&1; then
  echo "Installing system build dependencies (may ask for sudo)..."
  sudo apt update
  sudo apt install -y build-essential python3-dev libssl-dev libffi-dev curl ca-certificates || true
else
  echo "No apt found; ensure build toolchain and python dev headers are installed as needed."
fi
echo ""

# Set default for INSTALL_DEV if not already set (for non-interactive mode)
INSTALL_DEV=${INSTALL_DEV:-0}

# Install development tools (act, docker) if dev environment is requested
if [ "${INSTALL_DEV}" = "1" ]; then
  echo "==> Installing Development Tools (act, Docker)"
  
  # Install act for local CI testing
  if command -v act >/dev/null 2>&1; then
    echo "✓ act already installed: $(command -v act)"
  else
    echo "Installing act for local CI testing..."
    if command -v curl >/dev/null 2>&1; then
      curl -sSfL https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
      if command -v act >/dev/null 2>&1; then
        echo "✓ act installed successfully: $(command -v act)"
      else
        echo "⚠️  act installation completed but not found in PATH. Try: hash -r"
      fi
    else
      echo "⚠️  curl not available, cannot install act automatically."
      echo "Please install act manually: https://github.com/nektos/act#installation"
    fi
  fi
  
  # Check Docker installation
  if command -v docker >/dev/null 2>&1; then
    echo "✓ Docker already installed: $(command -v docker)"
  else
    echo "Docker not found. act requires Docker to run."
    if command -v apt >/dev/null 2>&1; then
      read -r -p "Install Docker via apt? [y/N]: " _docker_ans
      case "$_docker_ans" in
        [Yy]* )
          echo "Installing Docker..."
          sudo apt install -y docker.io
          sudo systemctl start docker || true
          sudo systemctl enable docker || true
          # Add current user to docker group
          sudo usermod -aG docker "$USER" || true
          echo "Docker installed. You may need to log out and back in for group changes to take effect."
          ;;
        * )
          echo "Skipping Docker installation. Install manually if needed for local CI."
          ;;
      esac
    else
      echo "⚠️  apt not available. Please install Docker manually: https://docs.docker.com/engine/install/"
    fi
  fi
  echo ""
else
  echo "==> Skipping Development Environment"
  echo "Running in minimal mode (no dev packages, act, or Docker)"
  echo "To install full dev environment later, run: INSTALL_DEV=1 ./scripts/setup_env.sh"
  echo ""
fi

echo "==> Installing UV Package Manager"
if command -v uv >/dev/null 2>&1; then
  echo "uv already available: $(command -v uv)"
else
  echo "uv not found; installing via: $UV_INSTALL_METHOD"
  case "$UV_INSTALL_METHOD" in
    skip)
      echo "Skipping uv installation (already installed)"
      ;;
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
echo ""

echo "==> Configuring Python Dependencies"

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
echo ""

echo "==> Syncing Python Environment"
if [ "${INSTALL_DEV}" = "1" ] || [ "${INSTALL_DEV}" = "yes" ]; then
  echo "Including development dependency group..."
  uv sync --all-groups
else
  echo "Skipping development dependency group (dev)."
  uv sync --no-dev
fi
echo ""

echo "==> Installing Pre-commit Hooks"
if [ "${INSTALL_DEV}" = "1" ]; then
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
echo ""

echo "==> Running Validation Checks"
if [ "${RUN_TESTS:-0}" = "1" ]; then
  echo "Running linters and tests..."
  uv run ruff check . || true
  uv run mypy src tests || true
  uv run pytest -q || true
  echo ""
else
  echo "Skipping validation checks (RUN_TESTS not set)"
  echo "To run checks later: RUN_TESTS=1 ./scripts/setup_env.sh"
  echo ""
fi

# Optional: Run local CI if RUN_LOCAL_CI is set
if [ "${RUN_LOCAL_CI:-0}" = "1" ]; then
  echo "==> Running Local CI with act"
  if command -v act >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
    "${SCRIPT_DIR}/run_ci_locally.sh" || {
      echo "Local CI failed. Fix issues before pushing."
      exit 1
    }
  else
    echo "⚠️  act or Docker not installed. Skipping local CI run."
    echo "To run CI locally, install Docker and act:"
    echo "  sudo apt install docker.io -y"
    echo "  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
  fi
  echo ""
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              ✓ Environment Setup Complete!                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
if [ "${INSTALL_DEV}" = "1" ]; then
  echo "  • Activate the environment: source .venv/bin/activate"
  echo "  • Run tests: uv run pytest"
  echo "  • Run linters: uv run ruff check ."
  echo "  • Run type checking: uv run mypy src tests"
  echo "  • Run CI locally: ./scripts/run_ci_locally.sh"
else
  echo "  • Activate the environment: source .venv/bin/activate"
  echo "  • Run your application: uv run python -m neural_analysis"
  echo "  • To install dev tools: INSTALL_DEV=1 ./scripts/setup_env.sh"
fi
echo ""
echo "Happy coding! 🚀"
echo "" 
