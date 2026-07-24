"""Super simple check: prepare inputs with the openai/whisper-large-v3 processor."""

import numpy as np

from transformers import WhisperProcessor


processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# 3 seconds of dummy mono audio at 16 kHz
sampling_rate = 16000
audio = np.random.randn(3 * sampling_rate).astype("float32")

inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")

for key, value in inputs.items():
    print(key, tuple(value.shape), value.dtype)

