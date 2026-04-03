"""Quick smoke test: start a serve instance with Gemma 4 and verify audio + video work."""

import base64
import socket
import time

import httpx
from openai import OpenAI

from transformers.cli.serve import Serve

AUDIO_URL = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3"
VIDEO_URL = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/concert.mp4"
MODEL = "google/gemma-4-E2B-it"


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_serve():
    port = find_free_port()
    serve = Serve(port=port, non_blocking=True)
    for _ in range(30):
        try:
            if httpx.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                return serve, port
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Server did not start in time")


serve, port = start_serve()
client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="unused")

audio_bytes = httpx.get(AUDIO_URL, follow_redirects=True).content
audio_b64 = base64.b64encode(audio_bytes).decode()

# --- Test 1: Audio (chat completions) ---
print("=== Test 1: Audio via chat completions ===")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ],
    max_tokens=200,
)
print(resp.choices[0].message.content)
print()

# --- Test 2: Video with audio (chat completions) ---
print("=== Test 2: Video via chat completions ===")
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": VIDEO_URL}},
                {"type": "text", "text": "Transcribe the lyrics of the song being played in this video."},
            ],
        }
    ],
    max_tokens=500,
)
print(resp.choices[0].message.content)
print()

# --- Test 3: Audio (responses API) ---
print("=== Test 3: Audio via responses API ===")
resp = client.responses.create(
    model=MODEL,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
            ],
        }
    ],
    stream=False,
    max_output_tokens=200,
)
print(resp.output[0].content[0].text)
print()

# --- Test 4: Video with audio (responses API) ---
print("=== Test 4: Video via responses API ===")
resp = client.responses.create(
    model=MODEL,
    input=[
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": VIDEO_URL}},
                {"type": "text", "text": "Transcribe the lyrics of the song being played in this video."},
            ],
        }
    ],
    stream=False,
    max_output_tokens=500,
)
print(resp.output[0].content[0].text)
print()

serve.kill_server()
print("Done!")
