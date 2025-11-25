#!/bin/bash
# Development helper script for Docker container

set -e

# Use docker compose (v2) if available, otherwise docker-compose (v1)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check if containers are running
if ! $DOCKER_COMPOSE ps | grep -q "Up"; then
    echo "Containers are not running. Starting them..."
    $DOCKER_COMPOSE up -d
    sleep 3
fi

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments: enter interactive shell
    echo "Entering development container..."
    $DOCKER_COMPOSE exec app bash
elif [ "$1" == "test" ]; then
    # Run tests
    echo "Running tests..."
    shift
    $DOCKER_COMPOSE exec app pytest "$@"
elif [ "$1" == "jupyter" ]; then
    # Start Jupyter notebook
    echo "Starting Jupyter notebook..."
    $DOCKER_COMPOSE exec -d app jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    echo "Jupyter notebook started at http://localhost:8888"
elif [ "$1" == "redis-cli" ]; then
    # Connect to Redis CLI
    echo "Connecting to Redis CLI..."
    $DOCKER_COMPOSE exec app redis-cli
elif [ "$1" == "logs" ]; then
    # Show logs
    shift
    $DOCKER_COMPOSE logs -f "$@"
else
    # Execute arbitrary command
    $DOCKER_COMPOSE exec app "$@"
fi



