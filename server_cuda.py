"""
Twake Whisper Backend - CUDA Version (Hybrid)
FastAPI server for real-time speech-to-text
Supports both TheStageAI optimized models and OpenAI Whisper models via faster-whisper
"""

import base64
import os
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Try to import TheStageAI SDK first
THESTAGE_AVAILABLE = False
try:
    from thestage_speechkit.nvidia import ASRPipeline as TheStageASR
    THESTAGE_AVAILABLE = True
    print("✅ TheStageAI SDK loaded successfully")
except ImportError:
    print("ℹ️  TheStageAI SDK not available, will use faster-whisper")

# Try to import faster-whisper as fallback
FASTER_WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("✅ faster-whisper loaded successfully")
except ImportError:
    print("⚠️  faster-whisper not available")

if not THESTAGE_AVAILABLE and not FASTER_WHISPER_AVAILABLE:
    print("❌ No backend available! Please install either:")
    print("   - TheStageAI SDK: pip install thestage_speechkit[nvidia]")
    print("   - faster-whisper: pip install faster-whisper")


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
# Backend Interface
# ============================================================================

class WhisperBackend(ABC):
    """Abstract interface for Whisper backends"""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str, word_timestamps: bool = True) -> Tuple[List[Word], dict]:
        """Transcribe audio and return words with timestamps"""
        pass


class TheStageBackend(WhisperBackend):
    """TheStageAI optimized backend"""

    def __init__(self, model_name: str, device: str = "cuda", chunk_length_s: int = 10):
        self.model_name = model_name
        self.device = device
        self.chunk_length_s = chunk_length_s

        print(f"🔄 Loading TheStageAI model: {model_name}")
        self.model = TheStageASR(
            model=model_name,
            chunk_length_s=chunk_length_s,
            batch_size=32,
            device=device
        )
        print("✅ TheStageAI model loaded")

    def transcribe(self, audio: np.ndarray, language: str, word_timestamps: bool = True) -> Tuple[List[Word], dict]:
        """Transcribe using TheStageAI"""
        result = self.model(
            audio,
            generate_kwargs={
                'do_sample': False,
                'use_cache': True,
                'language': language,
                'return_timestamps': 'word' if word_timestamps else True
            }
        )

        words = []
        info = {
            'language': language,
            'text': result.get('text', '')
        }

        # Extract words from chunks
        if 'chunks' in result:
            for chunk in result['chunks']:
                if 'timestamp' in chunk:
                    words.append(Word(
                        text=chunk['text'].strip(),
                        timestamp=list(chunk['timestamp']) if isinstance(chunk['timestamp'], tuple) else chunk['timestamp']
                    ))
                else:
                    words.append(Word(text=chunk['text'].strip(), timestamp=None))
        elif result.get('text'):
            # No word-level timestamps, return full text
            words.append(Word(text=result['text'].strip(), timestamp=None))

        return words, info


class FasterWhisperBackend(WhisperBackend):
    """faster-whisper backend with OpenAI models"""

    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        print(f"🔄 Loading faster-whisper model: {model_name}")
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=os.getenv("MODEL_CACHE_DIR", "/models")
        )
        print("✅ faster-whisper model loaded")

    def transcribe(self, audio: np.ndarray, language: str, word_timestamps: bool = True) -> Tuple[List[Word], dict]:
        """Transcribe using faster-whisper"""
        segments, info = self.model.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            ),
            beam_size=5,
            condition_on_previous_text=False
        )

        words = []
        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                for word_info in segment.words:
                    words.append(Word(
                        text=word_info.word.strip(),
                        timestamp=[word_info.start, word_info.end]
                    ))
            else:
                text = segment.text.strip()
                if text:
                    words.append(Word(
                        text=text,
                        timestamp=[segment.start, segment.end]
                    ))

        return words, {'language': info.language if hasattr(info, 'language') else language}


# ============================================================================
# Streaming Manager
# ============================================================================

class StreamingManager:
    """Manages concurrent transcription sessions with hybrid backend support"""

    def __init__(self, backend_type: str = "auto", model_name: str = None, device: str = "cuda",
                 compute_type: str = "float16", chunk_length_s: int = 10):
        self.sample_rate = 16000  # Whisper requires 16kHz
        self.sessions: Dict[str, dict] = {}
        self.chunk_duration = 5  # seconds - reduced for better UX
        self.uncommitted_threshold = 1.0  # seconds - minimum audio for uncommitted

        print(f"🎮 Device: {device}")
        print(f"🎤 Sample rate: {self.sample_rate}Hz")
        print(f"⏱️  Commit duration: {self.chunk_duration}s")
        print(f"⏱️  Uncommitted threshold: {self.uncommitted_threshold}s")

        # Initialize backend
        self.backend = self._init_backend(backend_type, model_name, device, compute_type, chunk_length_s)

        if not self.backend:
            raise RuntimeError("No Whisper backend available")

    def _init_backend(self, backend_type: str, model_name: Optional[str], device: str,
                      compute_type: str, chunk_length_s: int) -> Optional[WhisperBackend]:
        """Initialize the appropriate backend based on availability and preference"""

        # Auto-select backend
        if backend_type == "auto":
            if THESTAGE_AVAILABLE:
                backend_type = "thestage"
            elif FASTER_WHISPER_AVAILABLE:
                backend_type = "faster-whisper"
            else:
                return None

        print(f"📦 Backend: {backend_type}")

        if backend_type == "thestage":
            if not THESTAGE_AVAILABLE:
                print("❌ TheStageAI SDK not available, falling back to faster-whisper")
                backend_type = "faster-whisper"
            else:
                model = model_name or "TheStageAI/thewhisper-large-v3-turbo"
                print(f"📦 Model: {model}")
                return TheStageBackend(model, device, chunk_length_s)

        if backend_type == "faster-whisper":
            if not FASTER_WHISPER_AVAILABLE:
                print("❌ faster-whisper not available")
                return None
            else:
                model = model_name or "large-v3-turbo"
                print(f"📦 Model: {model}")
                print(f"🔢 Compute type: {compute_type}")
                return FasterWhisperBackend(model, device, compute_type)

        return None

    def create_session(self, language: str = "en") -> str:
        """Create a new transcription session"""
        if not self.backend:
            raise RuntimeError("No backend available")

        session_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode('utf-8').rstrip('=')

        self.sessions[session_id] = {
            "language": language,
            "active": True,
            "audio_buffer": np.array([], dtype=np.float32),
            "transcribed_text": [],
            "committed_words": []
        }

        print(f"✅ Session created: {session_id} (language: {language})")
        return session_id

    def add_chunk(self, session_id: str, audio_base64: str):
        """Add audio chunk to session buffer"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        session = self.sessions[session_id]
        session["audio_buffer"] = np.concatenate([session["audio_buffer"], audio_array])

    def process(self, session_id: str) -> TranscriptionResponse:
        """Get current transcription results"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]
        audio_buffer = session["audio_buffer"]
        language = session["language"]

        buffer_length_s = len(audio_buffer) / self.sample_rate

        committed = session["committed_words"].copy()
        uncommitted = []

        # Process committed transcription
        if buffer_length_s >= self.chunk_duration:
            try:
                new_words, info = self.backend.transcribe(audio_buffer, language, word_timestamps=True)

                session["committed_words"].extend(new_words)
                committed = session["committed_words"].copy()

                full_text = " ".join([w.text for w in new_words])
                if full_text:
                    session["transcribed_text"].append(full_text)

                session["audio_buffer"] = np.array([], dtype=np.float32)
                print(f"✅ Committed {buffer_length_s:.1f}s: {full_text[:50]}...")

            except Exception as e:
                print(f"⚠️ Transcription error: {e}")
                import traceback
                traceback.print_exc()

        # Process uncommitted transcription
        elif buffer_length_s >= self.uncommitted_threshold:
            try:
                uncommitted, info = self.backend.transcribe(audio_buffer, language, word_timestamps=True)

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

            if len(session["audio_buffer"]) > 0:
                try:
                    words, info = self.backend.transcribe(
                        session["audio_buffer"],
                        session["language"],
                        word_timestamps=False
                    )

                    final_text = " ".join([w.text for w in words])
                    if final_text:
                        session["transcribed_text"].append(final_text)
                        print(f"✅ Final transcription: {final_text[:50]}...")
                except Exception as e:
                    print(f"⚠️ Final transcription error: {e}")

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

manager: Optional[StreamingManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global manager

    print("=" * 60)
    print("🚀 Starting Twake Whisper Backend (Hybrid CUDA)")
    print("=" * 60)

    backend_type = os.getenv("BACKEND_TYPE", "auto")
    model_name = os.getenv("MODEL_NAME", None)
    device = os.getenv("DEVICE", "cuda")
    compute_type = os.getenv("COMPUTE_TYPE", "float16")
    chunk_length_s = int(os.getenv("CHUNK_LENGTH_S", "10"))

    try:
        manager = StreamingManager(
            backend_type=backend_type,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            chunk_length_s=chunk_length_s
        )
        print(f"✅ Backend ready")
    except Exception as e:
        print(f"⚠️  Failed to initialize manager: {e}")
        import traceback
        traceback.print_exc()
        manager = None

    print("=" * 60)

    yield

    print("\n🛑 Shutting down Twake Whisper Backend...")
    if manager:
        for session_id in list(manager.sessions.keys()):
            manager.end_session(session_id)


app = FastAPI(
    title="TheWhisper-api (Hybrid CUDA)",
    description="""
## Real-time Speech-to-Text API with Hybrid Whisper Support

Supports both:
- **TheStageAI optimized models** (thewhisper-large-v3-turbo) - requires thestage_speechkit
- **OpenAI Whisper models** (large-v3-turbo) - via faster-whisper

### Features
- 🚀 **Hybrid Backend**: Auto-detects available backend (TheStageAI or faster-whisper)
- 🎯 **Session-based Streaming**: Multiple concurrent transcription sessions
- ⏱️ **Word Timestamps**: Precise word-level timing information
- 🔄 **Real-time Processing**: Low-latency transcription
- 🌐 **OpenAI Compatible**: `/v1/audio/transcriptions` endpoint

### Documentation
- **Swagger UI**: [/docs](/docs)
- **ReDoc**: [/redoc](/redoc)
    """,
    version="1.0.0-hybrid",
    contact={
        "name": "TheWhisper-api",
        "url": "https://github.com/mmaudet/TheWhisper-api",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://twake-assistant.cozy.localhost:8080",
        "http://app.cozy.localhost:8080",
        "*"
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
    backend_info = "TheStageAI" if THESTAGE_AVAILABLE else "faster-whisper" if FASTER_WHISPER_AVAILABLE else "None"
    return {
        "message": "TheWhisper-api - Hybrid CUDA Speech-to-Text",
        "status": "running",
        "platform": "CUDA/NVIDIA GPU",
        "backend": backend_info,
        "thestage_available": THESTAGE_AVAILABLE,
        "faster_whisper_available": FASTER_WHISPER_AVAILABLE,
        "version": "1.0.0-hybrid"
    }


@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if manager else "unavailable",
        "backend": "TheStageAI" if THESTAGE_AVAILABLE else "faster-whisper" if FASTER_WHISPER_AVAILABLE else "None",
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


@app.post("/v1/audio/transcriptions", tags=["OpenAI Compatible"])
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0)
):
    """Create transcription (OpenAI compatible)"""
    if not manager:
        raise HTTPException(status_code=503, detail="Service not available")

    try:
        audio_bytes = await file.read()

        try:
            import librosa
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {str(e)}")

        words, info = manager.backend.transcribe(
            audio_array,
            language=language or "en",
            word_timestamps=(response_format == "verbose_json")
        )

        text = " ".join([w.text for w in words])

        if response_format == "text":
            return text
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": info.get('language', language or "en"),
                "duration": len(audio_array) / 16000.0,
                "text": text,
                "words": [{"word": w.text, "start": w.timestamp[0] if w.timestamp else 0,
                          "end": w.timestamp[1] if w.timestamp else 0} for w in words]
            }
        else:
            return {"text": text}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
