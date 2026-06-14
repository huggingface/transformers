<!--Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
-->
*This model was published in HF papers on 2025-09-15 and contributed to Hugging Face Transformers on 2026-06-14.*

# Fun-ASR-Nano

## Overview

Fun-ASR-Nano is an 800M-parameter end-to-end speech recognition model developed by Alibaba DAMO Academy's FunAudioLLM team. It achieves state-of-the-art performance on Chinese, English, and Japanese ASR benchmarks while being significantly smaller than comparable models.

The model was proposed in [Fun-ASR: An Industrial-Grade Speech Recognition System](https://huggingface.co/papers/2509.12508).

### Architecture

Fun-ASR-Nano consists of four components:

1. **Audio Encoder** (SenseVoiceEncoderSmall): A 70-layer SANM (Self-Attention with FSMN Memory) encoder that combines multi-head self-attention with Feedforward Sequential Memory Networks for efficient speech feature extraction.

2. **Audio Adaptor**: A 2-layer Transformer that projects encoder outputs (512-dim) to the LLM dimension (1024-dim).

3. **Language Model** (Qwen3-0.6B): A 28-layer causal language model that generates transcription text autoregressively.

4. **CTC Decoder** (optional): A 5-layer Transformer for character-level timestamp prediction via CTC forced alignment.

### Key Features

- **31 languages**: Chinese (including 7 dialects and 26 regional accents), English, Japanese, and 20+ European languages
- **Character-level timestamps** via CTC forced alignment
- **Hotword customization** for domain-specific vocabulary
- **Native punctuation** output (no separate punctuation model needed)
- **Streaming support** for chunk-by-chunk inference

## Usage

```python
from transformers import FunAsrNanoForConditionalGeneration, FunAsrNanoProcessor
import torch
import librosa

# Load model and processor
model = FunAsrNanoForConditionalGeneration.from_pretrained("FunAudioLLM/Fun-ASR-Nano-2512-hf")
processor = FunAsrNanoProcessor.from_pretrained("FunAudioLLM/Fun-ASR-Nano-2512-hf")

# Load audio
audio, sr = librosa.load("audio.wav", sr=16000)

# Prepare inputs

# Generate
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=200)

# Decode
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

## FunAsrNanoConfig

[[autodoc]] FunAsrNanoConfig

## FunAsrNanoEncoderConfig

[[autodoc]] FunAsrNanoEncoderConfig

## FunAsrNanoAdaptorConfig

[[autodoc]] FunAsrNanoAdaptorConfig

## FunAsrNanoCtcConfig

[[autodoc]] FunAsrNanoCtcConfig

## FunAsrNanoFeatureExtractor

[[autodoc]] FunAsrNanoFeatureExtractor
    - __call__

## FunAsrNanoProcessor

[[autodoc]] FunAsrNanoProcessor
    - __call__

## FunAsrNanoEncoder

[[autodoc]] FunAsrNanoEncoder
    - forward

## FunAsrNanoForConditionalGeneration

[[autodoc]] FunAsrNanoForConditionalGeneration
    - forward
