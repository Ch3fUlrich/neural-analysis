# Setup Script Usage Guide

The `scripts/setup_env.sh` script provides flexible environment setup with both **interactive** and **non-interactive** modes.

## Quick Start

### Interactive Mode (Recommended for First-Time Setup)

Simply run the script without any environment variables:

```bash
./scripts/setup_env.sh
```

The script will guide you through a menu to choose:
1. UV package manager installation method
2. Whether to install full development environment (Python packages, act, Docker)
3. Whether to run validation checks after setup

### Non-Interactive Mode (For Automation/CI)

Set environment variables to control the installation:

```bash
# Install full development environment
INSTALL_DEV=1 ./scripts/setup_env.sh

# Minimal installation (no dev tools)
INSTALL_DEV=0 ./scripts/setup_env.sh

# Full dev setup with validation
INSTALL_DEV=1 RUN_TESTS=1 ./scripts/setup_env.sh
```

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `UV_INSTALL_METHOD` | `installer`, `pipx`, `pip` | `installer` | How to install uv package manager |
| `INSTALL_DEV` | `0`, `1` | Prompt | Install full development environment (packages, act, Docker) |
| `RUN_TESTS` | `0`, `1` | `0` | Run linters and tests after setup to validate installation |
| `UV_FORCE_LOCK` | `0`, `1` | `0` | Force regeneration of uv.lock file |
| `RUN_LOCAL_CI` | `0`, `1` | `0` | Run local CI with act after setup |

## Common Usage Examples

### Fresh Development Setup

```bash
# Interactive - recommended for first time
./scripts/setup_env.sh
```

### CI/Build Server Setup

```bash
# Non-interactive, install dev packages for testing
INSTALL_DEV=1 UV_INSTALL_METHOD=installer ./scripts/setup_env.sh
```

### Minimal Production Setup

```bash
# No dev tools, just core dependencies
INSTALL_DEV=0 ./scripts/setup_env.sh
```

### Full Setup with Validation

```bash
# Install everything and run tests
INSTALL_DEV=1 RUN_TESTS=1 ./scripts/setup_env.sh
```

### Update Dependencies Only

```bash
# Assuming everything is installed, just sync packages
INSTALL_DEV=1 ./scripts/setup_env.sh
```

## What Gets Installed

### Base Installation (Always)
- System dependencies (build-essential, python3-dev, libssl-dev, libffi-dev, curl, ca-certificates)
- UV package manager (via chosen method)
- Core Python dependencies (from pyproject.toml)

### With `INSTALL_DEV=1` (Full Development Environment)
- **Python Development Packages:**
  - pytest, pytest-cov, pytest-xdist
  - mypy (type checking)
  - ruff (linting and formatting)
  - pre-commit (git hooks)
  - flake8, pydocstyle
  - coverage, build

- **Development Tools:**
  - **act** - Run GitHub Actions workflows locally (installed automatically)
  - **Docker** - Container runtime (required by act)
    - On Ubuntu/Debian with apt: offers to install docker.io
    - Adds current user to docker group
    - Starts and enables docker service

- **Pre-commit Hooks:**
  - Installed if inside a git repository
  - Runs quality checks before each commit

## Interactive Mode Features

When run without environment variables, the script provides:

1. **Clear Visual Interface**: Sectioned prompts with box-drawing characters
2. **Smart Defaults**: Sensible defaults for each choice (usually "yes")
3. **Configuration Summary**: Review all choices before proceeding
4. **Confirmation Step**: Final confirmation before starting installation
5. **Progress Indicators**: Clear section headers showing current step
6. **Success Message**: Helpful next steps after completion

## Post-Setup

After successful setup, the script displays next steps:

### With Development Environment (`INSTALL_DEV=1`):
```
╔════════════════════════════════════════════════════════════════╗
║              ✓ Environment Setup Complete!                    ║
╚════════════════════════════════════════════════════════════════╝

Next steps:
  • Activate the environment: source .venv/bin/activate
  • Run tests: uv run pytest
  • Run linters: uv run ruff check .
  • Run type checking: uv run mypy src tests
  • Run CI locally: ./scripts/run_ci_locally.sh

Happy coding! 🚀
```

### Minimal Installation (`INSTALL_DEV=0`):
```
Next steps:
  • Activate the environment: source .venv/bin/activate
  • Run your application: uv run python -m neural_analysis
  • To install dev tools: INSTALL_DEV=1 ./scripts/setup_env.sh
```

## Troubleshooting

### act Installation Fails

If act installation fails:
```bash
# Manual installation
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Or download from releases
wget https://github.com/nektos/act/releases/latest/download/act_Linux_x86_64.tar.gz
tar xzf act_Linux_x86_64.tar.gz
sudo mv act /usr/local/bin/
```

After installation, verify:
```bash
act --version
hash -r  # Refresh shell hash table if needed
```

### Docker Permission Issues

After Docker installation, you may need to log out and back in for group changes to take effect:
```bash
# Check if you're in the docker group
groups

# If not, the script already added you, just log out/in
# Or start a new shell with:
newgrp docker
```

Test Docker without sudo:
```bash
docker ps
```

### UV Not Found After Installation

If uv is not found after installation:
```bash
# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Make permanent by adding to your shell config
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # or ~/.zshrc

# Or restart your shell
exec $SHELL
```

### Pre-commit Hooks Not Installing

Pre-commit hooks require a git repository:
```bash
# Ensure you're in a git repository
git rev-parse --is-inside-work-tree

# If not, initialize git first
git init
```

### Script Exits Early

If the script exits unexpectedly:
```bash
# Run with bash tracing for debugging
bash -x ./scripts/setup_env.sh
```

### WSL Hardlink Warnings

On WSL, you may see warnings about hardlinking:
```
warning: Failed to hardlink files; falling back to full copy.
```

This is harmless. To suppress it:
```bash
export UV_LINK_MODE=copy
./scripts/setup_env.sh
```

## Advanced Usage

### Force Lockfile Regeneration

```bash
# Force regeneration of uv.lock even if it exists
UV_FORCE_LOCK=1 INSTALL_DEV=1 ./scripts/setup_env.sh
```

### Run Local CI After Setup

```bash
# Install dev environment and immediately run CI checks
INSTALL_DEV=1 RUN_LOCAL_CI=1 ./scripts/setup_env.sh
```

### Custom UV Installation

```bash
# Use pipx instead of the default installer
UV_INSTALL_METHOD=pipx INSTALL_DEV=1 ./scripts/setup_env.sh
```

## Integration with Development Workflow

### First-Time Setup
```bash
# Clone repository
git clone https://github.com/Ch3fUlrich/neural-analysis.git
cd neural-analysis

# Run interactive setup
./scripts/setup_env.sh
# Answer: Yes to dev environment, Yes to validation

# Start coding!
```

### Regular Development
```bash
# Activate environment
source .venv/bin/activate

# Make changes
# ...

# Run quality checks before pushing (MANDATORY per claude.md)
./scripts/run_ci_locally.sh

# Or manually:
uv run ruff check src tests
uv run mypy src tests
uv run pytest -v
```

### Updating Dependencies
```bash
# Update lockfile and sync
UV_FORCE_LOCK=1 INSTALL_DEV=1 ./scripts/setup_env.sh
```

## See Also

- [Local CI Testing Guide](local_ci_testing.md) - How to run GitHub Actions locally with act
- [Python Testing Setup](python_testing_setup.md) - Testing best practices
- [Contributing Guide](../CONTRIBUTING.md) - Development workflow
- [Claude.md](claude.md) - AI assistant guidelines (includes setup workflow)
- [README.md](../README.md) - Project overview and quick start

## Script Features Summary

✅ **Idempotent** - Safe to run multiple times
✅ **Interactive & Non-Interactive** - Works in CI and manual setups
✅ **Automatic act Installation** - No manual steps needed for local CI
✅ **Smart Detection** - Checks for existing installations
✅ **Clear Feedback** - Progress indicators and success messages
✅ **Error Handling** - Graceful fallbacks and helpful error messages
✅ **Cross-Platform** - Works on Ubuntu, Debian, WSL
