# Deployment Guide - CUDA Version

This guide explains how to deploy TheWhisper-api on a Linux server with NVIDIA GPU using Docker.

## Prerequisites

### Hardware Requirements
- Linux server with NVIDIA GPU (any CUDA-compatible GPU)
- Minimum 8GB GPU VRAM for `large-v3-turbo` model
- Minimum 4GB system RAM

### Software Requirements
1. **Docker** (20.10 or later)
2. **Docker Compose** (v2.0 or later)
3. **NVIDIA Container Toolkit**

## Installation

### 1. Install Docker

```bash
# Update package list
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in for group changes to take effect
```

### 2. Install NVIDIA Container Toolkit

```bash
# Configure repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker

# Test NVIDIA Docker integration
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information displayed.

### 3. Clone the Repository

```bash
git clone https://github.com/mmaudet/TheWhisper-api.git
cd TheWhisper-api
```

### 4. Configure Environment

```bash
# Copy the CUDA environment configuration
cp .env.cuda .env

# Edit configuration if needed
nano .env
```

Available models (trade-off between speed and accuracy):
- `tiny` - Fastest, least accurate (~1GB VRAM)
- `base` - Fast, moderate accuracy (~1GB VRAM)
- `small` - Balanced (~2GB VRAM)
- `medium` - Good accuracy (~5GB VRAM)
- `large-v2` - Very good accuracy (~10GB VRAM)
- `large-v3` - Best accuracy (~10GB VRAM)
- `large-v3-turbo` - Best balance (default, ~6GB VRAM)

Compute types:
- `float16` - Best balance (default)
- `int8_float16` - Faster, slightly less accurate
- `int8` - Fastest, reduced accuracy

## Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the container
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps

# Stop the service
docker compose down
```

### Option 2: Docker Run

```bash
# Build the image
docker build -t thewhisper-api:cuda .

# Run the container
docker run -d \
  --name whisper-api-cuda \
  --gpus all \
  -p 8000:8000 \
  -v whisper-models:/models \
  -e MODEL_NAME=large-v3-turbo \
  -e DEVICE=cuda \
  -e COMPUTE_TYPE=float16 \
  --restart unless-stopped \
  thewhisper-api:cuda

# View logs
docker logs -f whisper-api-cuda

# Stop the container
docker stop whisper-api-cuda
docker rm whisper-api-cuda
```

### First Run

The first time you start the service, it will download the Whisper model (~1-6GB depending on the model size). This may take several minutes depending on your internet connection.

```bash
# Monitor the download progress
docker compose logs -f
```

Once you see:
```
✅ Model loaded successfully
✅ Backend ready on CUDA with faster-whisper
```

The service is ready to accept requests.

## Testing the Deployment

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "cuda_available": true,
  "active_sessions": 0
}
```

### Create Session and Test Transcription

```bash
# Create a session
SESSION_ID=$(curl -s -X POST http://localhost:8000/session/create/ \
  -H "Content-Type: application/json" \
  -d '{"language": "en"}' | jq -r '.session_id')

echo "Session ID: $SESSION_ID"

# Test with a simple audio file (you'll need to provide your own audio)
# The audio should be Float32 PCM, 16kHz, mono, base64 encoded
```

### OpenAI Compatible API Test

```bash
# Test with an audio file
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@your_audio.mp3 \
  -F model=whisper-1 \
  -F language=en
```

## Monitoring

### Check GPU Usage

```bash
# Inside the container
docker exec whisper-api-cuda nvidia-smi

# Or from the host
nvidia-smi
```

### View Logs

```bash
# Real-time logs
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100

# Logs for specific time period
docker compose logs --since 1h
```

### Performance Metrics

```bash
# Container resource usage
docker stats whisper-api-cuda
```

## Updating

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# If this fails, reinstall NVIDIA Container Toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory Errors

- Use a smaller model (e.g., `small` or `base` instead of `large-v3-turbo`)
- Use `int8` compute type instead of `float16`
- Reduce concurrent sessions

### Model Download Fails

```bash
# Clear model cache and restart
docker compose down
docker volume rm thewhisper-api_whisper-models
docker compose up -d
```

### Port Already in Use

```bash
# Check what's using port 8000
sudo lsof -i :8000

# Change port in docker-compose.yml
# Edit the ports section: "8080:8000" instead of "8000:8000"
```

## Production Considerations

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name whisper-api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for long transcriptions
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

### SSL/TLS with Let's Encrypt

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d whisper-api.yourdomain.com
```

### Firewall Configuration

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# If accessing directly (without reverse proxy)
sudo ufw allow 8000/tcp
```

### Automatic Restarts

The Docker Compose configuration includes `restart: unless-stopped`, which automatically restarts the container on system reboot or if it crashes.

### Backup Model Cache

```bash
# Backup models volume
docker run --rm -v thewhisper-api_whisper-models:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/whisper-models-backup.tar.gz -C /data .

# Restore models volume
docker run --rm -v thewhisper-api_whisper-models:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/whisper-models-backup.tar.gz -C /data
```

## Resource Requirements by Model

| Model | VRAM | Speed (RTX 3090) | Accuracy |
|-------|------|------------------|----------|
| tiny | ~1GB | ~50x realtime | Low |
| base | ~1GB | ~40x realtime | Moderate |
| small | ~2GB | ~20x realtime | Good |
| medium | ~5GB | ~10x realtime | Very Good |
| large-v2 | ~10GB | ~5x realtime | Excellent |
| large-v3 | ~10GB | ~5x realtime | Excellent |
| large-v3-turbo | ~6GB | ~8x realtime | Excellent |

*Speeds are approximate and vary by GPU model and audio content*

## Support

For issues and questions:
- GitHub Issues: https://github.com/mmaudet/TheWhisper-api/issues
- Check logs: `docker compose logs -f`
- Verify GPU: `nvidia-smi`
