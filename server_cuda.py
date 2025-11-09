"""
Twake Whisper Backend - CUDA Version
FastAPI server for real-time speech-to-text using faster-whisper on NVIDIA GPUs
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

# Faster-Whisper imports
try:
    from faster_whisper import WhisperModel
    CUDA_AVAILABLE = True
    print("✅ faster-whisper loaded successfully")
except ImportError as e:
    print(f"⚠️  faster-whisper not available: {e}")
    print("Please install: pip install faster-whisper")
    CUDA_AVAILABLE = False


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
    """Manages concurrent transcription sessions using faster-whisper"""

    def __init__(self, model_name: str = "large-v3-turbo", device: str = "cuda", compute_type: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = 16000  # Whisper requires 16kHz
        self.sessions: Dict[str, dict] = {}
        self.chunk_duration = 5  # seconds - reduced for better UX
        self.uncommitted_threshold = 1.0  # seconds - minimum audio for uncommitted

        print(f"📦 Model: {model_name}")
        print(f"🎮 Device: {device}")
        print(f"🔢 Compute type: {compute_type}")
        print(f"🎤 Sample rate: {self.sample_rate}Hz")
        print(f"⏱️  Commit duration: {self.chunk_duration}s")
        print(f"⏱️  Uncommitted threshold: {self.uncommitted_threshold}s")

        # Initialize model
        if CUDA_AVAILABLE:
            print("🔄 Loading Whisper model...")
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=os.getenv("MODEL_CACHE_DIR", "/models")
            )
            print("✅ Model loaded successfully")
        else:
            self.model = None

    def create_session(self, language: str = "en") -> str:
        """Create a new transcription session"""
        if not CUDA_AVAILABLE:
            raise RuntimeError("faster-whisper not available")

        session_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=')

        self.sessions[session_id] = {
            "language": language,
            "active": True,
            "audio_buffer": np.array([], dtype=np.float32),
            "transcribed_text": [],  # All committed transcriptions
            "last_result": None,
            "committed_words": []
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
        committed = session["committed_words"].copy()

        # If we have enough audio for final transcription (>= chunk_duration)
        if buffer_length_s >= self.chunk_duration:
            try:
                # Transcribe with faster-whisper
                segments, info = self.model.transcribe(
                    audio_buffer,
                    language=language,
                    word_timestamps=True,
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=100
                    ),
                    beam_size=5,
                    condition_on_previous_text=False
                )

                # Extract new committed words from segments
                new_committed = []
                for segment in segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word_info in segment.words:
                            new_committed.append(Word(
                                text=word_info.word.strip(),
                                timestamp=[word_info.start, word_info.end]
                            ))
                    else:
                        # No word timestamps, use segment text
                        text = segment.text.strip()
                        if text:
                            new_committed.append(Word(
                                text=text,
                                timestamp=[segment.start, segment.end]
                            ))

                # Add to committed words
                session["committed_words"].extend(new_committed)
                committed = session["committed_words"].copy()

                # Store transcribed text
                full_text = " ".join([w.text for w in new_committed])
                if full_text:
                    session["transcribed_text"].append(full_text)

                # Clear buffer after transcription
                session["audio_buffer"] = np.array([], dtype=np.float32)

                print(f"✅ Committed {buffer_length_s:.1f}s: {full_text[:50]}...")

            except Exception as e:
                print(f"⚠️ Transcription error: {e}")
                import traceback
                traceback.print_exc()
        elif buffer_length_s >= self.uncommitted_threshold:
            # Have some audio but not enough for final - transcribe as uncommitted
            try:
                segments, info = self.model.transcribe(
                    audio_buffer,
                    language=language,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.5,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=100
                    ),
                    beam_size=5,
                    condition_on_previous_text=False
                )

                # Extract uncommitted words
                for segment in segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word_info in segment.words:
                            uncommitted.append(Word(
                                text=word_info.word.strip(),
                                timestamp=[word_info.start, word_info.end]
                            ))
                    else:
                        text = segment.text.strip()
                        if text:
                            uncommitted.append(Word(
                                text=text,
                                timestamp=[segment.start, segment.end]
                            ))

                if uncommitted:
                    full_text = " ".join([w.text for w in uncommitted])
                    print(f"🔄 Uncommitted {buffer_length_s:.1f}s: {full_text[:50]}...")

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
                    segments, info = self.model.transcribe(
                        session["audio_buffer"],
                        language=session["language"],
                        vad_filter=True
                    )

                    final_text = " ".join([segment.text.strip() for segment in segments])
                    if final_text:
                        session["transcribed_text"].append(final_text)
                        print(f"✅ Final transcription: {final_text[:50]}...")
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
        session["committed_words"] = []
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
    print("🚀 Starting Twake Whisper Backend (CUDA Version)")
    print("=" * 60)

    model_name = os.getenv("MODEL_NAME", "large-v3-turbo")
    device = os.getenv("DEVICE", "cuda")
    compute_type = os.getenv("COMPUTE_TYPE", "float16")

    if CUDA_AVAILABLE:
        try:
            manager = StreamingManager(
                model_name=model_name,
                device=device,
                compute_type=compute_type
            )
            print(f"✅ Backend ready on {device.upper()} with faster-whisper")
        except Exception as e:
            print(f"⚠️  Failed to initialize manager: {e}")
            import traceback
            traceback.print_exc()
            manager = None
    else:
        print("⚠️  faster-whisper not available - running in disabled mode")
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
    title="TheWhisper-api (CUDA)",
    description="""
## Real-time Speech-to-Text API with faster-whisper on NVIDIA GPUs

TheWhisper-api provides high-performance, real-time speech-to-text transcription using faster-whisper optimized for NVIDIA GPUs with CUDA.

### Features
- 🚀 **CUDA Acceleration**: Optimized for NVIDIA GPUs
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
    version="1.0.0-cuda",
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
        "http://app.cozy.localhost:8080",
        "*"  # Allow all origins for Docker deployment
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
    """Get API information"""
    return {
        "message": "TheWhisper-api - CUDA Speech-to-Text",
        "status": "running",
        "platform": "CUDA/NVIDIA GPU",
        "cuda_available": CUDA_AVAILABLE,
        "version": "1.0.0-cuda"
    }


@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cuda_available": CUDA_AVAILABLE,
        "active_sessions": len(manager.sessions) if manager else 0
    }


@app.post("/session/create/", response_model=SessionResponse, tags=["Session-based API"])
async def create_session(request: SessionCreateRequest):
    """Create a new transcription session"""
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        session_id = manager.create_session(language=request.language)
        return SessionResponse(session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/add_chunk", tags=["Session-based API"])
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


@app.post("/session/{session_id}/process", response_model=TranscriptionResponse, tags=["Session-based API"])
async def process_session(session_id: str):
    """Process and get transcription results"""
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
    """End transcription session"""
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        manager.end_session(session_id)
        return {"status": "ended"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/{session_id}/clear", tags=["Session-based API"])
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
    """Create transcription (OpenAI compatible)"""
    if not manager or not CUDA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        # Read audio file
        audio_bytes = await file.read()

        # Convert audio to numpy array
        try:
            import librosa
            # Load audio with librosa (handles various formats)
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {str(e)}")

        # Transcribe with faster-whisper
        segments, info = manager.model.transcribe(
            audio_array,
            language=language,
            word_timestamps=(response_format == "verbose_json"),
            vad_filter=True
        )

        # Convert segments generator to list
        segments_list = list(segments)

        # Extract text
        text = " ".join([segment.text.strip() for segment in segments_list])

        # Format response according to response_format
        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            # Return detailed format similar to OpenAI
            formatted_segments = []
            for i, seg in enumerate(segments_list):
                segment_data = {
                    "id": i,
                    "seek": 0,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "tokens": [],
                    "temperature": temperature,
                    "avg_logprob": seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0,
                    "compression_ratio": seg.compression_ratio if hasattr(seg, 'compression_ratio') else 0.0,
                    "no_speech_prob": seg.no_speech_prob if hasattr(seg, 'no_speech_prob') else 0.0
                }
                if hasattr(seg, 'words') and seg.words:
                    segment_data["words"] = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability if hasattr(w, 'probability') else 1.0
                        }
                        for w in seg.words
                    ]
                formatted_segments.append(segment_data)

            return {
                "task": "transcribe",
                "language": info.language if hasattr(info, 'language') else (language or "en"),
                "duration": len(audio_array) / 16000.0,
                "text": text,
                "segments": formatted_segments
            }
        else:  # json (default)
            return {"text": text}

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
