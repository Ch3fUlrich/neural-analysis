#!/bin/bash
# Setup script for Docker development environment

set -e

echo "Setting up Docker development environment for neural-analysis..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Use docker compose (v2) if available, otherwise docker-compose (v1)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "Building Docker images..."
$DOCKER_COMPOSE build

echo "Starting containers..."
$DOCKER_COMPOSE up -d

echo "Waiting for services to be ready..."
sleep 5

# Check Redis health
echo "Checking Redis connection..."
if $DOCKER_COMPOSE exec -T redis redis-cli ping &> /dev/null || \
   $DOCKER_COMPOSE exec -T app redis-cli ping &> /dev/null; then
    echo "✓ Redis is running"
else
    echo "⚠ Warning: Redis may not be ready yet"
fi

echo ""
echo "Setup complete! Your development environment is ready."
echo ""
echo "To enter the development container, run:"
echo "  $DOCKER_COMPOSE exec app bash"
echo ""
echo "Or use the helper script:"
echo "  ./scripts/dev_docker.sh"
echo ""
echo "To stop the containers, run:"
echo "  $DOCKER_COMPOSE down"
echo ""
echo "To view logs, run:"
echo "  $DOCKER_COMPOSE logs -f"



