"""
Twake Whisper Backend - MLX Native
FastAPI server for real-time speech-to-text using MLX-Whisper on Apple Silicon
"""

import base64
import os
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# MLX Whisper imports
try:
    import mlx.core as mx
    from mlx_whisper import transcribe
    MLX_AVAILABLE = True
    print("✅ MLX-Whisper loaded successfully")
except ImportError as e:
    print(f"⚠️  MLX-Whisper not available: {e}")
    print("Please install: pip install mlx mlx-whisper")
    MLX_AVAILABLE = False


# ============================================================================
# Models
# ============================================================================

class SessionCreateRequest(BaseModel):
    language: str = "en"


class SessionResponse(BaseModel):
    session_id: str


class Word(BaseModel):
    text: str
    timestamp: Optional[List[float]] = None


class TranscriptionResponse(BaseModel):
    committed: List[Word]
    uncommitted: List[Word]


class AudioChunkRequest(BaseModel):
    audio_base64: str


# ============================================================================
# Streaming Manager
# ============================================================================

class StreamingManager:
    """Manages concurrent transcription sessions using MLX-Whisper"""

    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        self.model_name = model_name
        self.sample_rate = 16000  # Whisper requires 16kHz
        self.sessions: Dict[str, dict] = {}
        self.chunk_duration = 5  # seconds - reduced for better UX
        self.uncommitted_threshold = 1.0  # seconds - minimum audio for uncommitted

        print(f"📦 Model: {model_name}")
        print(f"🎤 Sample rate: {self.sample_rate}Hz")
        print(f"⏱️  Commit duration: {self.chunk_duration}s")
        print(f"⏱️  Uncommitted threshold: {self.uncommitted_threshold}s")

    def create_session(self, language: str = "en") -> str:
        """Create a new transcription session"""
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX-Whisper not available")

        session_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=')

        self.sessions[session_id] = {
            "language": language,
            "active": True,
            "audio_buffer": np.array([], dtype=np.float32),
            "transcribed_text": [],  # All committed transcriptions
            "last_result": None
        }

        print(f"✅ Session created: {session_id} (language: {language})")
        return session_id

    def add_chunk(self, session_id: str, audio_base64: str):
        """Add audio chunk to session buffer"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        # Decode base64 audio (Float32 PCM)
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Append to buffer
        session = self.sessions[session_id]
        session["audio_buffer"] = np.concatenate([session["audio_buffer"], audio_array])

    def process(self, session_id: str) -> TranscriptionResponse:
        """Get current transcription results"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]
        audio_buffer = session["audio_buffer"]
        language = session["language"]

        # Calculate buffer length in seconds
        buffer_length_s = len(audio_buffer) / self.sample_rate

        committed = []
        uncommitted = []

        # Always return previously committed words
        if "committed_words" not in session:
            session["committed_words"] = []

        committed = session["committed_words"].copy()

        # If we have enough audio for final transcription (>= chunk_duration)
        if buffer_length_s >= self.chunk_duration:
            try:
                # Transcribe with MLX-Whisper
                result = transcribe(
                    audio_buffer,
                    path_or_hf_repo=self.model_name,
                    language=language,
                    word_timestamps=True,
                    verbose=False,
                    no_speech_threshold=0.8,  # Higher = more strict against silence hallucinations
                    logprob_threshold=-0.8,   # Higher (less negative) = less hallucinations
                    compression_ratio_threshold=2.0  # Lower = stricter against repetitions
                )

                # Extract new committed words from segments
                new_committed = []
                if "segments" in result:
                    for segment in result["segments"]:
                        if "words" in segment and segment["words"]:
                            for word_info in segment["words"]:
                                new_committed.append(Word(
                                    text=word_info["word"].strip(),
                                    timestamp=[word_info["start"], word_info["end"]]
                                ))
                        else:
                            # No word timestamps, use segment text
                            text = segment["text"].strip()
                            if text:
                                new_committed.append(Word(
                                    text=text,
                                    timestamp=[segment["start"], segment["end"]]
                                ))

                # Add to committed words
                session["committed_words"].extend(new_committed)
                committed = session["committed_words"].copy()

                # Store transcribed text
                if result.get("text"):
                    session["transcribed_text"].append(result["text"].strip())

                # Clear buffer after transcription
                session["audio_buffer"] = np.array([], dtype=np.float32)

                print(f"✅ Committed {buffer_length_s:.1f}s: {result.get('text', '')[:50]}...")

            except Exception as e:
                print(f"⚠️ Transcription error: {e}")
                import traceback
                traceback.print_exc()
        elif buffer_length_s >= self.uncommitted_threshold:
            # Have some audio but not enough for final - transcribe as uncommitted
            try:
                result = transcribe(
                    audio_buffer,
                    path_or_hf_repo=self.model_name,
                    language=language,
                    word_timestamps=True,
                    verbose=False,
                    no_speech_threshold=0.8,  # Higher = more strict against silence hallucinations
                    logprob_threshold=-0.8,   # Higher (less negative) = less hallucinations
                    compression_ratio_threshold=2.0  # Lower = stricter against repetitions
                )

                # Extract uncommitted words
                if "segments" in result:
                    for segment in result["segments"]:
                        if "words" in segment and segment["words"]:
                            for word_info in segment["words"]:
                                uncommitted.append(Word(
                                    text=word_info["word"].strip(),
                                    timestamp=[word_info["start"], word_info["end"]]
                                ))
                        else:
                            text = segment["text"].strip()
                            if text:
                                uncommitted.append(Word(
                                    text=text,
                                    timestamp=[segment["start"], segment["end"]]
                                ))

                if uncommitted:
                    print(f"🔄 Uncommitted {buffer_length_s:.1f}s: {result.get('text', '')[:50]}...")

            except Exception as e:
                print(f"⚠️ Uncommitted transcription error: {e}")

        return TranscriptionResponse(committed=committed, uncommitted=uncommitted)

    def end_session(self, session_id: str):
        """End and cleanup session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]

            # Final transcription of any remaining audio
            if len(session["audio_buffer"]) > 0:
                try:
                    result = transcribe(
                        session["audio_buffer"],
                        path_or_hf_repo=self.model_name,
                        language=session["language"],
                        verbose=False
                    )
                    if result.get("text"):
                        session["transcribed_text"].append(result["text"].strip())
                        print(f"✅ Final transcription: {result['text'][:50]}...")
                except Exception as e:
                    print(f"⚠️ Final transcription error: {e}")

            # Get final text
            final_text = " ".join(session["transcribed_text"])
            print(f"📝 Session {session_id} total: {len(final_text)} chars")

            del self.sessions[session_id]
            print(f"✅ Session ended: {session_id}")

    def clear_session(self, session_id: str):
        """Clear session buffers"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]
        session["audio_buffer"] = np.array([], dtype=np.float32)
        session["transcribed_text"] = []
        print(f"🗑️  Session cleared: {session_id}")


# ============================================================================
# FastAPI Application
# ============================================================================

# Global manager instance
manager: Optional[StreamingManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global manager

    # Startup
    print("=" * 60)
    print("🚀 Starting Twake Whisper Backend (MLX Native)")
    print("=" * 60)

    model_name = os.getenv("MODEL_NAME", "mlx-community/whisper-large-v3-turbo")

    if MLX_AVAILABLE:
        try:
            manager = StreamingManager(model_name=model_name)
            print(f"✅ Backend ready on Apple Silicon with MLX")
        except Exception as e:
            print(f"⚠️  Failed to initialize manager: {e}")
            manager = None
    else:
        print("⚠️  MLX not available - running in disabled mode")
        manager = None

    print("=" * 60)

    yield

    # Shutdown
    print("\n🛑 Shutting down Twake Whisper Backend...")
    if manager:
        # End all active sessions
        for session_id in list(manager.sessions.keys()):
            manager.end_session(session_id)


# Create FastAPI app
app = FastAPI(
    title="TheWhisper-api",
    description="""
## Real-time Speech-to-Text API with MLX-Whisper on Apple Silicon

TheWhisper-api provides high-performance, real-time speech-to-text transcription using MLX-Whisper optimized for Apple Silicon.

### Features
- 🚀 **MLX Acceleration**: Optimized for Apple Silicon (M1/M2/M3) GPUs
- 🎯 **Session-based Streaming**: Manage multiple concurrent transcription sessions
- ⏱️ **Word Timestamps**: Precise word-level timing information
- 🔄 **Real-time Processing**: Low-latency transcription with configurable chunk sizes
- 🌐 **OpenAI Compatible**: Supports OpenAI Audio API format

### API Formats
- **Session-based API**: `/session/*` endpoints for streaming transcription
- **OpenAI Compatible**: `/v1/audio/transcriptions` for drop-in replacement

### Documentation
- **Swagger UI**: [/docs](/docs)
- **ReDoc**: [/redoc](/redoc)
    """,
    version="1.0.0",
    contact={
        "name": "TheWhisper-api",
        "url": "https://github.com/mmaudet/TheWhisper-api",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://twake-assistant.cozy.localhost:8080",
        "http://app.cozy.localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """
    Get API information

    Returns basic information about the API, including platform and MLX availability status.
    """
    return {
        "message": "TheWhisper-api - MLX Native Speech-to-Text",
        "status": "running",
        "platform": "Apple Silicon",
        "mlx_available": MLX_AVAILABLE,
        "version": "1.0.0"
    }


@app.get("/health", tags=["General"])
async def health():
    """
    Health check endpoint

    Check if the service is healthy and get current status including:
    - Overall health status
    - MLX availability
    - Number of active transcription sessions
    """
    return {
        "status": "healthy",
        "mlx_available": MLX_AVAILABLE,
        "active_sessions": len(manager.sessions) if manager else 0
    }


@app.post("/session/create/", response_model=SessionResponse, tags=["Session-based API"])
async def create_session(request: SessionCreateRequest):
    """
    Create a new transcription session

    Initializes a new real-time transcription session with the specified language.
    Returns a unique session ID for subsequent operations.

    **Parameters:**
    - **language**: Language code (e.g., 'en', 'fr', 'es'). Default: 'en'

    **Returns:**
    - **session_id**: URL-safe unique identifier for this session

    **Example:**
    ```json
    {
      "language": "en"
    }
    ```
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        session_id = manager.create_session(language=request.language)
        return SessionResponse(session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/add_chunk", tags=["Session-based API"])
async def add_chunk(session_id: str, request: AudioChunkRequest):
    """
    Add audio chunk to session

    Stream audio data to an active transcription session. Audio should be:
    - Format: Float32 PCM
    - Sample rate: 16kHz
    - Channels: Mono
    - Encoding: Base64

    **Parameters:**
    - **session_id**: The session ID from create_session
    - **audio_base64**: Base64-encoded Float32 PCM audio data

    **Returns:**
    - **status**: Operation result ('success' or error)
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        manager.add_chunk(session_id, request.audio_base64)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/process", response_model=TranscriptionResponse, tags=["Session-based API"])
async def process_session(session_id: str):
    """
    Process and get transcription results

    Trigger transcription processing and retrieve results. Transcription begins when
    the audio buffer reaches the configured chunk duration (default: 15 seconds).

    **Parameters:**
    - **session_id**: The session ID from create_session

    **Returns:**
    - **committed**: List of finalized transcribed words with timestamps
    - **uncommitted**: List of in-progress words (currently unused)

    **Note:** This endpoint processes accumulated audio and returns all committed transcriptions.
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        result = manager.process(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/end", tags=["Session-based API"])
async def end_session(session_id: str):
    """
    End transcription session

    Finalizes and cleans up a transcription session. Any remaining audio in the buffer
    will be processed before the session is closed.

    **Parameters:**
    - **session_id**: The session ID to end

    **Returns:**
    - **status**: 'ended' on success
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        manager.end_session(session_id)
        return {"status": "ended"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/clear", tags=["Session-based API"])
async def clear_session(session_id: str):
    """
    Clear session buffers

    Resets the audio buffer and transcription history for an active session without ending it.
    Useful for starting a new transcription segment within the same session.

    **Parameters:**
    - **session_id**: The session ID to clear

    **Returns:**
    - **status**: 'cleared' on success
    """
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        manager.clear_session(session_id)
        return {"status": "cleared"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# OpenAI Compatible API
# ============================================================================

@app.post("/v1/audio/transcriptions", tags=["OpenAI Compatible"])
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: str = Form(default="whisper-1", description="Model to use for transcription"),
    language: Optional[str] = Form(default=None, description="Language of the audio (ISO-639-1)"),
    response_format: str = Form(default="json", description="Format of the response: json, text, srt, verbose_json, vtt"),
    temperature: float = Form(default=0.0, description="Sampling temperature (0-1)")
):
    """
    Create transcription (OpenAI compatible)

    Transcribes audio into the input language. This endpoint is compatible with OpenAI's
    Audio API, allowing drop-in replacement for applications using OpenAI's Whisper API.

    **Compatible with**: OpenAI Audio API v1

    **Supported formats**: mp3, mp4, mpeg, mpga, m4a, wav, webm

    **Parameters:**
    - **file**: Audio file to transcribe (required)
    - **model**: Model ID (default: "whisper-1", ignored - uses MLX-Whisper)
    - **language**: Language code (e.g., 'en', 'fr', 'es')
    - **response_format**: Output format (json, text, verbose_json)
    - **temperature**: Sampling temperature 0-1 (currently ignored)

    **Returns:**
    - **json**: `{"text": "transcribed text"}`
    - **text**: Plain text transcription
    - **verbose_json**: Detailed JSON with segments and timestamps

    **Example:**
    ```bash
    curl -X POST http://localhost:8000/v1/audio/transcriptions \\
      -F file=@audio.mp3 \\
      -F model=whisper-1 \\
      -F language=en
    ```
    """
    if not manager or not MLX_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Convert audio to numpy array
        # For simplicity, assume WAV/PCM format or use librosa for conversion
        try:
            import librosa
            # Load audio with librosa (handles various formats)
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {str(e)}")

        # Transcribe with MLX-Whisper
        from mlx_whisper import transcribe as mlx_transcribe

        result = mlx_transcribe(
            audio_array,
            path_or_hf_repo=manager.model_name,
            language=language,
            word_timestamps=(response_format == "verbose_json"),
            verbose=False
        )

        # Format response according to response_format
        if response_format == "text":
            return result.get("text", "")
        elif response_format == "verbose_json":
            # Return detailed format similar to OpenAI
            segments = []
            if "segments" in result:
                for seg in result["segments"]:
                    segment_data = {
                        "id": seg.get("id", 0),
                        "seek": seg.get("seek", 0),
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", ""),
                        "tokens": seg.get("tokens", []),
                        "temperature": temperature,
                        "avg_logprob": seg.get("avg_logprob", 0.0),
                        "compression_ratio": seg.get("compression_ratio", 0.0),
                        "no_speech_prob": seg.get("no_speech_prob", 0.0)
                    }
                    if "words" in seg:
                        segment_data["words"] = seg["words"]
                    segments.append(segment_data)

            return {
                "task": "transcribe",
                "language": result.get("language", language or "en"),
                "duration": len(audio_array) / 16000.0,
                "text": result.get("text", ""),
                "segments": segments
            }
        else:  # json (default)
            return {"text": result.get("text", "")}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Signal Handlers
# ============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"\n🚀 Starting server on port {port}...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
