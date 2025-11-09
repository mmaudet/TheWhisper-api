# TheWhisper-api (MLX Native)

Real-time speech-to-text API backend using MLX-Whisper on Apple Silicon.

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.11+**
- **MLX framework** (only works natively on macOS)

## Installation

### 1. Create Virtual Environment

```bash
cd /Users/mmaudet/work/TheWhisper-api
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- MLX and MLX-Whisper (Apple Silicon optimized)
- FastAPI and Uvicorn (web framework)
- NumPy and Librosa (audio processing)

### 3. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` if needed (optional):

```env
MODEL_NAME=mlx-community/whisper-large-v3-turbo
PORT=8000
```

## Running the Server

### Development Mode

```bash
python server.py
```

The server will start on `http://localhost:8000`

### Production Mode

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

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
3. Reinstall MLX:
   ```bash
   pip install --upgrade mlx mlx-whisper
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
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore         # Git ignore rules
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
