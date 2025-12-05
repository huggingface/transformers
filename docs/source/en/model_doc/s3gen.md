# S3Gen

## Overview

S3Gen is a complete text-to-speech model that converts speech tokens to waveforms using speaker-conditioned Conditional Flow Matching (CFM) and HiFTNet vocoder. It was introduced in the [CosyVoice2 paper](https://arxiv.org/abs/2409.15939) and is part of the [chatterbox](https://github.com/resemble-ai/chatterbox) TTS.

The model consists of four main components:

1. **S3 Tokenizer**: Tokenizes reference audio to extract speech tokens
2. **CAMPPlus Speaker Encoder**: Extracts speaker embeddings from reference audio
3. **CFM Flow Decoder**: Generates mel spectrograms from speech tokens using conditional flow matching
4. **HiFTNet Vocoder**: Converts mel spectrograms to waveforms

S3Gen enables zero-shot voice cloning by conditioning the generation on a reference audio sample. The model uses a causal architecture suitable for streaming applications.

## Model Architecture

The S3Gen model follows this pipeline:

```
Reference Audio → S3 Tokenizer + CAMPPlus → Speaker Embeddings
Speech Tokens + Speaker Embeddings → CFM Decoder → Mel Spectrograms
Mel Spectrograms → HiFTNet → Waveforms
```

### Key Features

- **Zero-shot voice cloning**: Clone any voice from a short reference audio sample
- **High-quality synthesis**: Uses conditional flow matching for natural mel spectrogram generation
- **Neural source-filter vocoder**: HiFTNet provides high-fidelity waveform synthesis
- **Causal architecture**: Supports streaming inference
- **Speaker conditioning**: Robust speaker embedding extraction via CAMPPlus encoder

## Usage

### Basic Usage

```python
from transformers import S3GenModel
import torch
import torchaudio

# Load model
model = S3GenModel.from_pretrained("ResembleAI/s3gen")
model.eval()

# Load reference audio
ref_wav, ref_sr = torchaudio.load("reference.wav")

# Create speech tokens (from your TTS frontend or S3 tokenizer)
speech_tokens = torch.randint(0, 6561, (1, 100))  # Example tokens

# Generate waveform
with torch.no_grad():
    waveform, _ = model.inference(
        speech_tokens=speech_tokens,
        ref_wav=ref_wav,
        ref_sr=ref_sr,
        finalize=True
    )

# Save output
torchaudio.save("output.wav", waveform.cpu(), 24000)
```

### Two-Stage Generation

You can also run the model in two stages for more control:

```python
# Stage 1: Generate mel spectrograms
with torch.no_grad():
    mel_spectrograms = model(
        speech_tokens=speech_tokens,
        ref_wav=ref_wav,
        ref_sr=ref_sr,
        finalize=True
    )

# Stage 2: Generate waveforms from mels
from transformers import HiFTNetModel

hiftnet = model.mel2wav  # or load separately
cache_source = torch.zeros(1, 1, 0)
with torch.no_grad():
    waveform, _ = hiftnet.inference(
        speech_feat=mel_spectrograms,
        cache_source=cache_source
    )
```

### Pre-computed Reference Embeddings

For production use, you can pre-compute reference embeddings:

```python
# Extract reference embeddings once
ref_dict = model.embed_ref(ref_wav, ref_sr)

# Reuse for multiple generations
with torch.no_grad():
    mel1 = model(tokens1, ref_dict=ref_dict, finalize=True)
    mel2 = model(tokens2, ref_dict=ref_dict, finalize=True)
```

## Model Details

### Input Requirements

- **Speech Tokens**: Integer tensor of shape `(batch_size, sequence_length)` with values in range `[0, 6560]`
- **Reference Audio**: Float tensor of shape `(batch_size, audio_length)` or `(audio_length,)`
- **Reference Sample Rate**: Integer (will be resampled internally to 16kHz for speaker encoder and 24kHz for mel extraction)

### Output

- **Mel Spectrograms**: Float tensor of shape `(batch_size, mel_bins, time_steps)` with `mel_bins=80`
- **Waveforms**: Float tensor of shape `(batch_size, audio_samples)` at 24kHz sample rate

### Configuration

The model can be configured via [`S3GenConfig`]:

```python
from transformers import S3GenConfig

config = S3GenConfig(
    vocab_size=6561,
    encoder_num_blocks=6,
    decoder_num_mid_blocks=12,
    sampling_rate=24000,
    mel_bins=80,
)
```

## Limitations

- Requires GPU for real-time performance
- Reference audio quality affects output quality
- Token sequence length affects generation time

## Citation

```bibtex
@article{cosyvoice2,
  title={CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models},
  author={Du, Zhihao and others},
  journal={arXiv preprint arXiv:2409.15939},
  year={2024}
}
```

## S3GenConfig

[[autodoc]] S3GenConfig

## HiFTNetConfig

[[autodoc]] HiFTNetConfig

## S3GenModel

[[autodoc]] S3GenModel
    - forward
    - inference
    - embed_ref

