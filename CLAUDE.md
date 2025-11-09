# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TheWhisper-api is a real-time speech-to-text API backend using MLX-Whisper optimized for Apple Silicon (M1/M2/M3). It provides both a session-based streaming API and an OpenAI-compatible transcription endpoint.

**Platform Requirements:**
- macOS with Apple Silicon only (MLX framework dependency)
- Python 3.11+
- Uses `uv` for dependency management (not pip)

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Create environment file (optional)
cp .env.example .env
```

### Running the Server
```bash
# Development mode (recommended)
uv run python server.py

# Alternative: activate venv first
source .venv/bin/activate
python server.py

# Production mode
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run the test script
uv run python test_backend.py

# Manual API testing
curl http://localhost:8000/health
```

## Architecture

### Core Components

**server.py (670 lines)** - Single-file application containing:
- `StreamingManager` (lines 66-259): Session-based transcription manager
  - Handles concurrent sessions with unique session IDs
  - Buffers audio chunks until chunk_duration threshold (default 5s)
  - Processes audio with MLX-Whisper when buffer is full
  - Returns "committed" words (finalized) and "uncommitted" words (in-progress)
- FastAPI application (lines 304-638): Web server with two API styles
- Signal handlers for graceful shutdown

### Session-Based Transcription Flow

1. **Create Session** → `/session/create/` returns session_id
2. **Stream Audio** → `/session/{session_id}/add_chunk` accumulates Float32 PCM audio (16kHz mono, base64 encoded)
3. **Process** → `/session/{session_id}/process` triggers transcription when buffer ≥ chunk_duration
4. **End** → `/session/{session_id}/end` finalizes and cleans up

Key behaviors:
- Audio buffer accumulates until reaching `chunk_duration` (5 seconds)
- Buffer is automatically cleared after successful transcription
- Committed words persist throughout session lifetime
- Final transcription processes any remaining audio on session end

### Audio Processing Details

- **Sample Rate**: 16kHz (Whisper requirement)
- **Format**: Float32 PCM mono, base64 encoded for transport
- **Buffer Management**: Numpy arrays concatenated in `audio_buffer`
- **Transcription Parameters** (lines 143-146, 193-196):
  - `no_speech_threshold=0.8` - Higher value reduces silence hallucinations
  - `logprob_threshold=-0.8` - Higher (less negative) reduces general hallucinations
  - `compression_ratio_threshold=2.0` - Lower value stricter against repetitions

### API Endpoints

**Session-based API** (lines 389-527):
- POST `/session/create/` - Initialize session with language
- POST `/session/{session_id}/add_chunk` - Add audio chunk (body: `{"audio_base64": "..."}`)
- POST `/session/{session_id}/process` - Get transcription results
- POST `/session/{session_id}/end` - Finalize session
- POST `/session/{session_id}/clear` - Reset buffers without ending

**OpenAI Compatible API** (lines 533-637):
- POST `/v1/audio/transcriptions` - Drop-in replacement for OpenAI Audio API
  - Accepts multipart form: file, model, language, response_format, temperature
  - Uses librosa to load various audio formats
  - Supports response_format: json, text, verbose_json

**General** (lines 356-386):
- GET `/` - API info
- GET `/health` - Service health and active session count

### Configuration

Environment variables (loaded from `.env` via python-dotenv):
- `MODEL_NAME` - MLX Whisper model (default: `mlx-community/whisper-large-v3-turbo`)
- `PORT` - Server port (default: 8000)

Available models:
- `mlx-community/whisper-base` (smallest, fastest)
- `mlx-community/whisper-small`
- `mlx-community/whisper-medium`
- `mlx-community/whisper-large-v3`
- `mlx-community/whisper-large-v3-turbo` (default, balanced)

### Dependencies (pyproject.toml)

Key dependencies:
- **MLX stack**: `mlx==0.25.2`, `mlx-whisper==0.4.0` (Apple Silicon GPU acceleration)
- **Web**: `fastapi==0.115.5`, `uvicorn[standard]==0.32.1`
- **Audio**: `librosa==0.10.2.post1`, `numpy==1.26.4`

Dev dependencies (in dependency-groups):
- pytest, pytest-asyncio, httpx

### CORS Configuration

Configured for Twake Assistant Cozy app (lines 339-349):
- `http://localhost:8080`
- `http://twake-assistant.cozy.localhost:8080`
- `http://app.cozy.localhost:8080`

### Model Download Behavior

First run downloads the Whisper model (~1-3GB depending on size) and caches it. Subsequent runs use the cached model.

## Important Implementation Notes

- **Single-file architecture**: All code is in `server.py` for simplicity
- **Lifespan management** (lines 269-302): Initializes `StreamingManager` on startup, cleans up sessions on shutdown
- **Session cleanup**: All active sessions are ended on application shutdown
- **Error handling**: MLX availability checked at startup; graceful degradation if unavailable
- **Thread safety**: Not explicitly handled - assumes single-threaded async operation
- **Memory management**: Sessions store audio buffers in memory; large models require more RAM
