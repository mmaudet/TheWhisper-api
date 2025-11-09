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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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
        self.chunk_duration = 15  # seconds

        print(f"📦 Model: {model_name}")
        print(f"🎤 Sample rate: {self.sample_rate}Hz")
        print(f"⏱️  Chunk duration: {self.chunk_duration}s")

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

        # If we have enough audio, transcribe
        if buffer_length_s >= self.chunk_duration:
            try:
                # Transcribe with MLX-Whisper
                result = transcribe(
                    audio_buffer,
                    path_or_hf_repo=self.model_name,
                    language=language,
                    word_timestamps=True,
                    verbose=False
                )

                # Extract words from segments
                if "segments" in result:
                    for segment in result["segments"]:
                        if "words" in segment and segment["words"]:
                            for word_info in segment["words"]:
                                committed.append(Word(
                                    text=word_info["word"].strip(),
                                    timestamp=[word_info["start"], word_info["end"]]
                                ))
                        else:
                            # No word timestamps, use segment text
                            text = segment["text"].strip()
                            if text:
                                committed.append(Word(
                                    text=text,
                                    timestamp=[segment["start"], segment["end"]]
                                ))

                # Store transcribed text
                if result.get("text"):
                    session["transcribed_text"].append(result["text"].strip())

                # Clear buffer after transcription
                session["audio_buffer"] = np.array([], dtype=np.float32)

                print(f"✅ Transcribed {buffer_length_s:.1f}s: {result.get('text', '')[:50]}...")

            except Exception as e:
                print(f"⚠️ Transcription error: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Not enough audio yet - return previous results
            for text in session["transcribed_text"]:
                committed.append(Word(text=text, timestamp=None))

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
    title="Twake Whisper Backend",
    description="Real-time speech-to-text with MLX-Whisper on Apple Silicon",
    version="1.0.0",
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Twake Whisper Backend - MLX Native",
        "status": "running",
        "platform": "Apple Silicon",
        "mlx_available": MLX_AVAILABLE
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mlx_available": MLX_AVAILABLE,
        "active_sessions": len(manager.sessions) if manager else 0
    }


@app.post("/session/create/", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new transcription session"""
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        session_id = manager.create_session(language=request.language)
        return SessionResponse(session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/add_chunk")
async def add_chunk(session_id: str, request: AudioChunkRequest):
    """Add audio chunk to session"""
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        manager.add_chunk(session_id, request.audio_base64)
        return {"status": "success"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/process", response_model=TranscriptionResponse)
async def process_session(session_id: str):
    """Get current transcription results"""
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        result = manager.process(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """End transcription session"""
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        manager.end_session(session_id)
        return {"status": "ended"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear session buffers"""
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
