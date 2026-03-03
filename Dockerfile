# Development Dockerfile for neural-analysis
# Includes Python environment, Redis, and development tools

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev,storage]"

# Copy source code
COPY . .

# Expose ports Redis
EXPOSE 6379 

# Start Redis in background and keep container running
CMD ["sh", "-c", "redis-server --daemonize yes && tail -f /dev/null"]



