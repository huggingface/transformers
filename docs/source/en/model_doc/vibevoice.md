<!--Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-01-20.*

# VibeVoice

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[VibeVoice](https://huggingface.co/microsoft/VibeVoice-1.5B) is a text-to-speech (TTS) model developed by Microsoft that generates high-fidelity, natural-sounding speech from text input. The model comes in two variants:

- **VibeVoice-1.5B**: A full-featured model for long-form speech generation with semantic tokenization and multi-speaker support.
- **VibeVoice-Realtime-0.5B**: A streaming variant optimized for real-time generation with ~300ms latency.

The model architecture consists of:

- A **Qwen2 language model backbone** for text understanding and generation
- An **acoustic tokenizer** (σ-VAE) that converts audio to/from continuous latent representations
- A **semantic encoder** (1.5B model only) for capturing high-level speech semantics
- A **diffusion head** for generating acoustic features from language model outputs
- **Speaker embeddings** support for voice cloning and multi-speaker synthesis

## Usage

### Basic Text-to-Speech

```python
>>> from transformers import VibeVoiceProcessor, VibeVoiceForConditionalGeneration
>>> import torch

>>> processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
>>> model = VibeVoiceForConditionalGeneration.from_pretrained("microsoft/VibeVoice-1.5B")

>>> text = "Hello, this is a test of the VibeVoice text-to-speech system."
>>> inputs = processor(text=text, return_tensors="pt")

>>> # Generate speech (implementation depends on full generation pipeline)
>>> outputs = model(**inputs)
```

### Streaming Model (Real-time)

```python
>>> from transformers import VibeVoiceStreamingForConditionalGeneration, AutoConfig

>>> model = VibeVoiceStreamingForConditionalGeneration.from_pretrained(
...     "microsoft/VibeVoice-Realtime-0.5B"
... )
>>> print(f"TTS backbone layers: {model.config.tts_backbone_num_hidden_layers}")
>>> print(f"Text backbone layers: {model.config.text_backbone_num_hidden_layers}")
```

### Using Speaker Embeddings

The VibeVoice processor supports speaker embeddings for voice cloning:

```python
>>> import numpy as np

>>> # Load with speaker embeddings
>>> processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")

>>> # Process with speaker embedding array
>>> speaker_embedding = np.random.randn(256).astype(np.float32)  # Example embedding
>>> inputs = processor(
...     text="Hello world",
...     speaker_embedding=speaker_embedding,
...     return_tensors="pt"
... )
```

### Using Half-Precision

You can reduce memory usage by loading the model in half-precision:

```python
>>> model = VibeVoiceForConditionalGeneration.from_pretrained(
...     "microsoft/VibeVoice-1.5B",
...     torch_dtype=torch.float16
... )
```

### Using Flash Attention 2

For faster inference, you can use Flash Attention 2:

```python
>>> model = VibeVoiceForConditionalGeneration.from_pretrained(
...     "microsoft/VibeVoice-1.5B",
...     torch_dtype=torch.float16,
...     attn_implementation="flash_attention_2"
... )
```

## Model Architecture

### VibeVoice-1.5B (Full Model)

The full model includes:
- `language_model`: Qwen2-based backbone (24 layers)
- `acoustic_tokenizer`: σ-VAE encoder and decoder for audio processing
- `semantic_tokenizer`: Encoder for high-level semantic features
- `prediction_head`: Diffusion-based head for acoustic generation
- `acoustic_connector` / `semantic_connector`: Feature projectors

### VibeVoice-Realtime-0.5B (Streaming Model)

The streaming model has a split architecture:
- `language_model`: Text backbone (4 layers)
- `tts_language_model`: TTS backbone (20 layers)
- `acoustic_tokenizer`: Decoder only (no encoder for streaming)
- `tts_eos_classifier`: End-of-speech classifier
- `tts_input_types`: Input type embeddings

## VibeVoiceConfig

[[autodoc]] VibeVoiceConfig
    - all

## VibeVoiceStreamingConfig

[[autodoc]] VibeVoiceStreamingConfig
    - all

## VibeVoiceAcousticCodecConfig

[[autodoc]] VibeVoiceAcousticCodecConfig
    - all

## VibeVoiceSemanticEncoderConfig

[[autodoc]] VibeVoiceSemanticEncoderConfig
    - all

## VibeVoiceDiffusionHeadConfig

[[autodoc]] VibeVoiceDiffusionHeadConfig
    - all

## VibeVoiceProcessor

[[autodoc]] VibeVoiceProcessor
    - __call__

## VibeVoiceModel

[[autodoc]] VibeVoiceModel
    - forward

## VibeVoiceForConditionalGeneration

[[autodoc]] VibeVoiceForConditionalGeneration
    - forward

## VibeVoiceStreamingForConditionalGeneration

[[autodoc]] VibeVoiceStreamingForConditionalGeneration
    - forward

## VibeVoiceStreamingModel

[[autodoc]] VibeVoiceStreamingModel
    - forward

## VibeVoiceAcousticCodec

[[autodoc]] VibeVoiceAcousticCodec
    - forward
    - encode
    - decode

## VibeVoiceSemanticEncoder

[[autodoc]] VibeVoiceSemanticEncoder
    - forward

## VibeVoiceDiffusionHead

[[autodoc]] VibeVoiceDiffusionHead
    - forward
