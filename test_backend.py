#!/usr/bin/env python3
"""
Test script for twake-whisper-backend
Tests the full audio transcription workflow
"""

import base64
import numpy as np
import requests
import time

BACKEND_URL = "http://localhost:8000"

def test_backend():
    print("🧪 Testing Twake Whisper Backend with MLX\n")

    # 1. Health check
    print("1️⃣ Health check...")
    response = requests.get(f"{BACKEND_URL}/health")
    health = response.json()
    print(f"   Status: {health['status']}")
    print(f"   MLX Available: {health['mlx_available']}")
    print(f"   Active Sessions: {health['active_sessions']}\n")

    if not health['mlx_available']:
        print("❌ MLX not available, aborting test")
        return

    # 2. Create session
    print("2️⃣ Creating session...")
    response = requests.post(f"{BACKEND_URL}/session/create/", json={"language": "en"})
    session_id = response.json()['session_id']
    print(f"   Session ID: {session_id}\n")

    # 3. Add audio chunks (silence for testing)
    print("3️⃣ Adding audio chunks (15 seconds of silence)...")
    # Create 1 second of silence (16kHz, Float32)
    audio = np.zeros(16000, dtype=np.float32)
    audio_base64 = base64.b64encode(audio.tobytes()).decode('utf-8')

    for i in range(15):
        response = requests.post(
            f"{BACKEND_URL}/session/{session_id}/add_chunk",
            json={"audio_base64": audio_base64}
        )
        if response.status_code == 200:
            print(f"   Chunk {i+1}/15 added ✓")
        else:
            print(f"   Chunk {i+1}/15 failed: {response.text}")
            return

    print()

    # 4. Process (trigger transcription)
    print("4️⃣ Processing transcription...")
    print("   (This may take a moment on first run as the model downloads...)")
    start_time = time.time()

    response = requests.post(f"{BACKEND_URL}/session/{session_id}/process")
    elapsed = time.time() - start_time

    result = response.json()
    print(f"   Processing took {elapsed:.2f} seconds")
    print(f"   Committed words: {len(result['committed'])}")
    print(f"   Uncommitted words: {len(result['uncommitted'])}")

    if result['committed']:
        print("\n   Transcription:")
        for word in result['committed'][:10]:  # Show first 10 words
            timestamp = word.get('timestamp', 'N/A')
            print(f"      - {word['text']} (timestamp: {timestamp})")
    else:
        print("   ℹ️  No transcription (silence detected)\n")

    # 5. End session
    print("\n5️⃣ Ending session...")
    response = requests.post(f"{BACKEND_URL}/session/{session_id}/end")
    print(f"   Session ended: {response.json()['status']}\n")

    print("✅ Backend test completed successfully!")

if __name__ == "__main__":
    try:
        test_backend()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
