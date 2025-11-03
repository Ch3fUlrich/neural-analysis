# Local CI Testing with act

This document explains how to run GitHub Actions workflows locally using `act`.

## Why Run CI Locally?

- **Faster feedback**: Catch issues before pushing to GitHub
- **Save CI minutes**: Reduce GitHub Actions usage
- **Offline testing**: Test workflows without internet access
- **Debugging**: Easier to debug workflow issues locally

## Prerequisites

1. **Docker**: Required to run the containerized workflows
2. **act**: Tool to run GitHub Actions locally

## Installation

### On Ubuntu/WSL

```bash
# Install Docker
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (optional, to avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### Verify Installation

```bash
# Check Docker
docker --version
docker ps

# Check act
act --version
```

## Usage

### Using the Provided Script (Recommended)

```bash
# Run the local CI script
./scripts/run_ci_locally.sh

# The script will:
# - Check if Docker and act are installed
# - Prompt for confirmation
# - Run the CI workflow locally
```

### Using act Directly

```bash
# Run the default workflow
act

# Run specific workflow
act -W .github/workflows/ci.yml

# Run with verbose output
act -W .github/workflows/ci.yml -v

# Run specific job
act -j test

# List available workflows
act -l

# Dry run (show what would be executed)
act -n
```

## Common Options

```bash
# Pull latest Docker images
act --pull

# Run without pulling images
act --pull=false

# Use specific platform
act --platform ubuntu-latest=catthehacker/ubuntu:act-latest

# Pass secrets
act --secret-file .secrets

# Pass environment variables
act --env VAR=value
```

## Troubleshooting

### Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo
sudo act
```

### act Not Found After Installation

```bash
# Ensure /usr/local/bin is in PATH
echo $PATH

# Or install to a different location
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash -s -- -b ~/.local/bin
```

### Workflow Fails Locally but Passes on GitHub

- Check Docker image compatibility
- Ensure all secrets/environment variables are set
- Some GitHub-specific features may not work identically locally

## Integration with Development Workflow

### Run Before Pushing

```bash
# Method 1: Use setup script with RUN_LOCAL_CI flag
RUN_LOCAL_CI=1 ./scripts/setup_env.sh

# Method 2: Run directly
./scripts/run_ci_locally.sh

# Method 3: Use act directly
act -W .github/workflows/ci.yml
```

### Pre-Push Hook (Optional)

Create `.git/hooks/pre-push`:

```bash
#!/bin/bash
echo "Running local CI checks before push..."
./scripts/run_ci_locally.sh
if [ $? -ne 0 ]; then
    echo "Local CI failed. Fix issues before pushing."
    exit 1
fi
```

Make it executable:

```bash
chmod +x .git/hooks/pre-push
```

## Performance Tips

1. **First run is slow**: Docker needs to download images (~1-2 GB)
2. **Use `--pull=false`**: Skip image updates for faster subsequent runs
3. **Clean old containers**: `docker system prune` to free space
4. **Use smaller images**: Consider `act -P ubuntu-latest=node:16-buster-slim` for faster startup

## Resources

- act documentation: https://github.com/nektos/act
- Docker documentation: https://docs.docker.com/
- GitHub Actions documentation: https://docs.github.com/actions
