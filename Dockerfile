# TheWhisper-api CUDA Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Stage 2: Dependencies
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-cuda.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-cuda.txt

# Stage 3: Final image
FROM base AS final

# Set working directory
WORKDIR /app

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages

# Copy application code
COPY server_cuda.py .
COPY .env.example .env

# Create models directory
RUN mkdir -p /models

# Set environment variables for the application
ENV MODEL_NAME=large-v3-turbo \
    DEVICE=cuda \
    COMPUTE_TYPE=float16 \
    PORT=8000 \
    MODEL_CACHE_DIR=/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "server_cuda.py"]
