# TheWhisper-api

Real-time speech-to-text API backend powered by Whisper.

## Two Versions Available

### 🍎 MLX Version (macOS Apple Silicon)
Optimized for Apple Silicon using MLX framework - runs locally on your MacBook with low power consumption.

**Best for:** Local development, privacy-focused applications, low-power requirements

### 🐳 CUDA Version (Linux + NVIDIA GPU)
Dockerized deployment for Linux servers with NVIDIA GPUs using faster-whisper.

**Best for:** Production deployments, high-throughput requirements, server environments

---

## MLX Version (Apple Silicon)

### Requirements

- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.11+**
- **MLX framework** (only works natively on macOS)

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

#### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via Homebrew:
```bash
brew install uv
```

#### 2. Install Dependencies

```bash
cd /Users/mmaudet/work/TheWhisper-api
uv sync
```

This creates a virtual environment (`.venv/`) and installs all dependencies:
- MLX and MLX-Whisper (Apple Silicon optimized)
- FastAPI and Uvicorn (web framework)
- NumPy and Librosa (audio processing)

#### 3. Configure Environment (Optional)

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` if needed:

```env
MODEL_NAME=mlx-community/whisper-large-v3-turbo
PORT=8000
```

### Running the Server

#### Development Mode

With uv (recommended):
```bash
uv run python server.py
```

Or activate the virtual environment first:
```bash
source .venv/bin/activate
python server.py
```

The server will start on `http://localhost:8000`

#### Production Mode

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## CUDA Version (Linux + NVIDIA GPU)

Supports **two Whisper backends**:
- **faster-whisper** (default): OpenAI models, open source, no licensing needed
- **TheStageAI SDK**: Optimized streaming models (thewhisper-large-v3-turbo), advanced features

The system **auto-detects** available backends and falls back gracefully.

### Requirements

- **Linux** server with NVIDIA GPU (any CUDA-compatible GPU)
- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit**
- Minimum 8GB GPU VRAM for `large-v3-turbo` model

### Quick Start

```bash
# Clone repository
git clone https://github.com/mmaudet/TheWhisper-api.git
cd TheWhisper-api

# Run deployment script
./deploy.sh
```

The script will:
1. Check prerequisites (Docker, NVIDIA Container Toolkit)
2. Create configuration from `.env.cuda`
3. Build Docker image
4. Start the service
5. Wait for health check

Service will be available at `http://localhost:8000`

### Pre-Built Images (Production)

**Recommended for production**: Use pre-built images from Docker Hub (no build time needed):

```bash
# Copy configuration
cp .env.cuda .env

# Pull and run pre-built images
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

**Available on Docker Hub:**
- 🐳 **Repository**: https://hub.docker.com/r/mmaudet/thewhisper-api
- `mmaudet/thewhisper-api:cuda` - faster-whisper (default)
- `mmaudet/thewhisper-api:latest` - Latest build
- Version tags: `1.0.0`, `1.0`, `1` for pinned deployments

**Note:** TheStageAI image (`thestage` tag) is not built in CI/CD due to disk space constraints (~3-4GB dependencies). Build locally if needed: `docker build -f Dockerfile.thestage .`

**Benefits:** ✅ No build time, ✅ CI/CD tested, ✅ Automatic updates

**Setup CI/CD:** See [DOCKER_HUB_SETUP.md](DOCKER_HUB_SETUP.md) to configure automated builds

See [CICD.md](CICD.md) for complete CI/CD documentation.

### Manual Deployment (Development)

Build locally for development or customization:

```bash
# Copy configuration
cp .env.cuda .env

# Build and start with Docker Compose
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### Configuration

Edit `.env` to customize:

```env
# Backend selection (auto, faster-whisper, thestage)
BACKEND_TYPE=auto

# Models:
# - OpenAI: tiny, base, small, medium, large-v3, large-v3-turbo
# - TheStageAI: TheStageAI/thewhisper-large-v3-turbo, TheStageAI/thewhisper-large-v3
MODEL_NAME=large-v3-turbo

# Device (cuda or cpu)
DEVICE=cuda

# Compute precision (float16, int8_float16, int8) - faster-whisper only
COMPUTE_TYPE=float16

# Chunk size (10, 15, 20, 30) - TheStageAI only
CHUNK_LENGTH_S=10
```

### Model Selection

**OpenAI Models** (faster-whisper):
| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~1GB | 50x realtime | Low |
| base | ~1GB | 40x realtime | Moderate |
| small | ~2GB | 20x realtime | Good |
| medium | ~5GB | 10x realtime | Very Good |
| large-v3-turbo | ~6GB | 8x realtime | Excellent ⭐ |
| large-v3 | ~10GB | 5x realtime | Excellent |

**TheStageAI Models**:
- `TheStageAI/thewhisper-large-v3-turbo` - Optimized for streaming (~6GB VRAM)
- `TheStageAI/thewhisper-large-v3` - Full optimized model (~10GB VRAM)

### Full Documentation

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide including:
- Detailed installation steps
- Production configurations
- Monitoring and troubleshooting
- SSL/TLS setup with Nginx
- Resource requirements

---

## API Endpoints

Both versions expose the same API endpoints.

### Health Check

```bash
curl http://localhost:8000/health
```

### Create Transcription Session

```bash
curl -X POST http://localhost:8000/session/create/ \
  -H "Content-Type: application/json" \
  -d '{"language": "en"}'
```

Response:
```json
{
  "session_id": "abc123..."
}
```

### Add Audio Chunk

```bash
curl -X POST "http://localhost:8000/session/{session_id}/add_chunk?base64={audio_base64}"
```

Audio format: Float32 PCM, 16kHz, mono, base64 encoded

### Process Session (Get Transcription)

```bash
curl -X POST http://localhost:8000/session/{session_id}/process
```

Response:
```json
{
  "committed": [
    {"text": "Hello", "timestamp": [0.0, 0.5]},
    {"text": "world", "timestamp": [0.5, 1.0]}
  ],
  "uncommitted": []
}
```

### End Session

```bash
curl -X POST http://localhost:8000/session/{session_id}/end
```

### Clear Session Buffers

```bash
curl -X POST http://localhost:8000/session/{session_id}/clear
```

## Architecture

### Session-Based Streaming

1. **Create Session**: Initialize a transcription session with language preference
2. **Add Chunks**: Stream audio chunks (Float32 PCM at 16kHz)
3. **Process**: Get transcription when buffer reaches chunk duration (15s default)
4. **End Session**: Cleanup and get final transcription

### Audio Processing

- **Sample Rate**: 16kHz (Whisper requirement)
- **Format**: Float32 PCM mono
- **Encoding**: Base64 for HTTP transmission
- **Buffer**: Accumulates audio until chunk_duration (15s)
- **Transcription**: Uses MLX-Whisper for Apple Silicon optimization

### Response Format

- **committed**: Finalized transcription with word timestamps
- **uncommitted**: In-progress transcription (empty in this implementation)

## Model Configuration

Default model: `mlx-community/whisper-large-v3-turbo`

Other available models:
- `mlx-community/whisper-base`
- `mlx-community/whisper-small`
- `mlx-community/whisper-medium`
- `mlx-community/whisper-large-v3`

Change model in `.env`:
```env
MODEL_NAME=mlx-community/whisper-base
```

## Performance

MLX-Whisper on Apple Silicon (M1/M2/M3):
- Very low power consumption (~2W)
- High throughput (~220 tokens/sec on M2)
- Real-time transcription capability

## Integration with Twake Assistant

This backend is designed to work with the Twake Assistant Cozy app:

1. **CORS**: Configured for Cozy domains (localhost:8080, cozy.localhost:8080)
2. **API**: Compatible with TheWhisper electron_app API
3. **Port**: Runs on 8000 (Cozy app on 8080)

Frontend connects via:
```javascript
const BACKEND_URL = 'http://localhost:8000';
```

## Troubleshooting

### MLX Not Available

If you see "MLX-Whisper not available":

1. Verify you're on macOS with Apple Silicon
2. Check Python version (3.11+)
3. Reinstall dependencies:
   ```bash
   uv sync --reinstall
   ```

### Model Download

First run will download the Whisper model (~1-3GB depending on size). This is cached for future runs.

### Memory Issues

Large models (large-v3) require more RAM. If you encounter issues:
- Use a smaller model (base, small, medium)
- Reduce chunk_duration in server.py

## Development

### Code Structure

```
TheWhisper-api/
├── server.py           # FastAPI application
├── pyproject.toml      # Project metadata and dependencies (uv)
├── uv.lock            # Dependency lockfile (uv)
├── requirements.txt    # Python dependencies (legacy, for reference)
├── .env.example        # Environment template
├── .gitignore         # Git ignore rules
├── test_backend.py    # Integration test script
└── README.md          # This file
```

### Testing

Test with a simple audio file:

```python
import base64
import numpy as np
import requests

# Create 1 second of silence (16kHz, Float32)
audio = np.zeros(16000, dtype=np.float32)
audio_base64 = base64.b64encode(audio.tobytes()).decode('utf-8')

# Create session
response = requests.post('http://localhost:8000/session/create/', json={'language': 'en'})
session_id = response.json()['session_id']

# Add audio chunks (15 seconds total for transcription)
for _ in range(15):
    requests.post(f'http://localhost:8000/session/{session_id}/add_chunk', json={'audio_base64': audio_base64})

# Process
result = requests.post(f'http://localhost:8000/session/{session_id}/process')
print(result.json())

# End session
requests.post(f'http://localhost:8000/session/{session_id}/end')
```

## About

TheWhisper-api is a standalone MLX-native backend for real-time speech-to-text transcription, designed to work with the Twake Assistant application. It leverages Apple Silicon's GPU acceleration through the MLX framework for optimal performance.

## References

- [TheWhisper GitHub](https://github.com/TheStageAI/TheWhisper)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [MLX-Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
