#!/usr/bin/env bash
set -e

# Local CI runner script using act (GitHub Actions locally)
# This script sets up and runs GitHub Actions workflows locally using Docker

echo "==> Local CI Runner for neural-analysis"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo ""
    echo "To install Docker on Ubuntu/WSL:"
    echo "  sudo apt update"
    echo "  sudo apt install docker.io -y"
    echo "  sudo systemctl start docker"
    echo "  sudo systemctl enable docker"
    echo ""
    echo "Add your user to docker group (optional, to avoid sudo):"
    echo "  sudo usermod -aG docker \$USER"
    echo "  newgrp docker"
    exit 1
fi

# Check if Docker daemon is running
if ! docker ps &> /dev/null; then
    echo "❌ Docker daemon is not running."
    echo ""
    echo "Start Docker with:"
    echo "  sudo systemctl start docker"
    echo ""
    echo "Check Docker status:"
    echo "  sudo systemctl status docker"
    exit 1
fi

echo "✅ Docker is installed and running"

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo "❌ act is not installed."
    echo ""
    echo "To install act:"
    echo "  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
    echo ""
    echo "Or manually:"
    echo "  wget https://github.com/nektos/act/releases/latest/download/act_Linux_x86_64.tar.gz"
    echo "  tar xzf act_Linux_x86_64.tar.gz"
    echo "  sudo mv act /usr/local/bin/"
    exit 1
fi

echo "✅ act is installed"
echo ""

# Determine which workflow to run
WORKFLOW="${1:-.github/workflows/ci.yml}"

echo "==> Running workflow: $WORKFLOW"
echo ""
echo "This will:"
echo "  1. Pull the GitHub Actions runner Docker image"
echo "  2. Set up the Python environment"
echo "  3. Install uv and dependencies"
echo "  4. Run linting, type checking, and tests"
echo ""
echo "Note: First run may take several minutes to download Docker images."
echo ""

# Run act with the specified workflow
# -W specifies the workflow file
# --pull=false prevents pulling images every time (remove for first run)
# Use --pull=true on first run or to update images

read -p "Continue? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "==> Executing act..."
echo ""

# Run act (add --pull=true for first run)
act -W "$WORKFLOW" "$@"

echo ""
echo "==> Local CI run complete!"
