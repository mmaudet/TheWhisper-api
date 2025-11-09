#!/bin/bash
# Quick deployment script for TheWhisper-api CUDA version

set -e

echo "======================================"
echo "TheWhisper-api CUDA Deployment Script"
echo "======================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker is installed"

# Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose v2."
    exit 1
fi

echo "✅ Docker Compose is installed"

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA Container Toolkit not detected or not configured properly"
    echo ""
    echo "To install NVIDIA Container Toolkit:"
    echo "1. Visit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "2. Or run: sudo apt-get install -y nvidia-container-toolkit"
    echo "3. Then restart Docker: sudo systemctl restart docker"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ NVIDIA Container Toolkit is configured"
fi

# Check if .env exists, if not copy from .env.cuda
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file from .env.cuda..."
    cp .env.cuda .env
    echo "✅ .env file created"
    echo ""
    echo "You can edit .env to customize the configuration:"
    echo "  - MODEL_NAME: Change model size (tiny, base, small, medium, large-v3-turbo)"
    echo "  - COMPUTE_TYPE: Adjust precision (float16, int8_float16, int8)"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to exit and edit .env first..."
fi

echo ""
echo "🏗️  Building Docker image..."
docker compose build

echo ""
echo "🚀 Starting TheWhisper-api..."
docker compose up -d

echo ""
echo "⏳ Waiting for service to start..."
sleep 5

# Wait for health check
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "✅ Service is healthy and ready!"
        break
    fi
    echo "⏳ Waiting for service to be ready... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo ""
    echo "⚠️  Service did not become healthy in time."
    echo "Check logs with: docker compose logs -f"
    exit 1
fi

echo ""
echo "======================================"
echo "🎉 Deployment Successful!"
echo "======================================"
echo ""
echo "Service is running at: http://localhost:8000"
echo ""
echo "📖 API Documentation:"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo ""
echo "🔍 Useful commands:"
echo "  - View logs: docker compose logs -f"
echo "  - Check status: docker compose ps"
echo "  - Stop service: docker compose down"
echo "  - Restart service: docker compose restart"
echo ""
echo "🧪 Test the API:"
echo "  curl http://localhost:8000/health"
echo ""
echo "Note: First run will download the model (~1-6GB)."
echo "Monitor progress with: docker compose logs -f"
echo ""
