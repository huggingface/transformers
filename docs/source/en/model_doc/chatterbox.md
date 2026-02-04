# Chatterbox

## Overview

Chatterbox is a complete text-to-speech (TTS) pipeline that combines three specialized models to convert text directly to natural-sounding speech. It was introduced in the [chatterbox repository](https://github.com/resemble-ai/chatterbox) and provides a unified interface for high-quality voice cloning and speech synthesis.

The pipeline consists of three main components:

1. **T3 Model**: Converts text tokens to speech tokens using a language model approach
2. **S3Gen Model**: Generates mel spectrograms from speech tokens using speaker-conditioned Conditional Flow Matching (CFM)
3. **HiFTNet Vocoder**: Converts mel spectrograms to high-fidelity waveforms

Chatterbox enables zero-shot voice cloning by conditioning the generation on a reference audio sample, allowing you to synthesize speech in any voice from just a few seconds of audio.

## Model Architecture

The Chatterbox model follows this end-to-end pipeline:

```
Text → Text Tokenizer → Text Tokens
Text Tokens + Style → T3 → Speech Tokens
Speech Tokens + Reference Audio → S3Gen → Mel Spectrograms
Mel Spectrograms → HiFTNet → Waveforms
```

### Key Features

- **End-to-end TTS**: Complete pipeline from text to waveform in a single model
- **Zero-shot voice cloning**: Clone any voice from a short reference audio sample
- **Multilingual support**: Supports both English-only and multilingual configurations
- **High-quality synthesis**: Uses state-of-the-art conditional flow matching and neural vocoding
- **Flexible configuration**: Each component can be configured independently
- **Style control**: Optional style conditioning for expressive speech synthesis

## Usage

### Basic Text-to-Speech

```python
from transformers import ChatterboxModel
import torch
import torchaudio

# Load model
model = ChatterboxModel.from_pretrained("ResembleAI/chatterbox-hf")
model = model.to("cuda")  # or "cpu"
model.eval()

# Load text tokenizer
model.load_text_tokenizer("path/to/tokenizer.json")

# Load reference audio for voice cloning
ref_wav, ref_sr = torchaudio.load("reference.wav")

# Convert to mono if needed
if ref_wav.shape[0] > 1:
    ref_wav = ref_wav.mean(dim=0, keepdim=True)

# Convert to numpy array
ref_audio = ref_wav.squeeze().numpy()

# Generate speech from text
text = "Hello, this is a text-to-speech demo using Chatterbox."
waveform = model.generate(
    text=text,
    reference_wav=ref_audio,
    reference_sr=ref_sr,
    exaggeration=0.5,
    temperature=0.8,
    top_p=0.95,
    min_p=0.05,
    repetition_penalty=1.2,
    cfg_weight=0.5,
    max_new_tokens=1000,
)

# Save output
torchaudio.save("output.wav", waveform.cpu().unsqueeze(0), 24000)
```

### Advanced: Two-Stage Generation

For more control, you can prepare the conditionals and run stages separately:

```python
import numpy as np

# Prepare conditionals (speaker embeddings and prompts)
conds = model.prepare_conditionals(
    reference_wav=ref_audio,  # numpy array
    reference_sr=ref_sr,
    exaggeration=0.5
)

# Stage 1: Prepare text tokens
text_tokens = model.prepare_text_tokens(text)

# Stage 2: Generate speech tokens using T3
with torch.no_grad():
    speech_tokens = model.t3.inference(
        t3_cond=conds.t3,
        text_tokens=text_tokens[0],
        max_new_tokens=1000,
        temperature=0.8,
        top_p=0.95,
    )

# Stage 3: Generate waveform using S3Gen
with torch.no_grad():
    waveform, _ = model.s3gen.inference(
        speech_tokens=speech_tokens[0],
        ref_dict=conds.gen,
        finalize=True
    )
```

### Pre-computed Conditionals for Batch Generation

For production use where you're generating multiple utterances with the same voice:

```python
import numpy as np

# Prepare conditionals once
conds = model.prepare_conditionals(
    reference_wav=ref_audio,
    reference_sr=ref_sr,
    exaggeration=0.5
)

# Generate multiple outputs efficiently
texts = ["First sentence.", "Second sentence.", "Third sentence."]
waveforms = []

for text in texts:
    # Prepare text tokens
    text_tokens = model.prepare_text_tokens(text)
    
    # Generate speech tokens
    with torch.no_grad():
        speech_tokens = model.t3.inference(
            t3_cond=conds.t3,
            text_tokens=text_tokens[0],
            max_new_tokens=1000,
            temperature=0.8,
        )
        
        # Generate waveform with cached embeddings
        waveform, _ = model.s3gen.inference(
            speech_tokens=speech_tokens[0],
            ref_dict=conds.gen,
            finalize=True
        )
        waveforms.append(waveform.squeeze(0))
```

### Generation with Return Intermediates

You can retrieve intermediate outputs (text tokens, speech tokens) for debugging:

```python
waveform, intermediates = model.generate(
    text=text,
    reference_wav=ref_audio,
    reference_sr=ref_sr,
    exaggeration=0.5,
    temperature=0.8,
    return_intermediates=True,
)

print(f"Text tokens: {intermediates['text_tokens'].shape}")
print(f"Speech tokens: {intermediates['speech_tokens'].shape}")
print(f"Waveform: {waveform.shape}")
```

## Model Details

### Configuration

The model can be configured via [`ChatterboxConfig`]:

```python
from transformers import ChatterboxConfig

# English-only configuration
config = ChatterboxConfig.english_only()

# Multilingual configuration
config = ChatterboxConfig.multilingual()

# Custom configuration
config = ChatterboxConfig(
    t3_config={"num_layers": 12, "num_heads": 16},
    s3gen_config={"encoder_num_blocks": 6},
    hiftnet_config={"upsample_rates": [5, 5, 4, 2]},
    is_multilingual=False,
)
```

### Input Requirements

For the `generate()` method:
- **Text**: String input (automatically normalized for punctuation via `punc_norm()`)
- **Reference Audio**: NumPy array of shape `(audio_length,)` - mono audio
- **Reference Sample Rate**: Integer (will be resampled internally to 16kHz for speaker encoder and 24kHz for mel extraction)
- **Tokenizer**: Optional - if not provided via parameter, must be loaded via `load_text_tokenizer()`

Generation parameters:
- **exaggeration** (float, default 0.5): Emotion/expressiveness level (0.0 to 1.0)
- **temperature** (float, default 0.8): Sampling temperature for token generation
- **top_p** (float, default 0.95): Top-p (nucleus) sampling threshold
- **min_p** (float, default 0.05): Minimum probability threshold
- **repetition_penalty** (float, default 1.2): Penalty for repeating tokens
- **cfg_weight** (float, default 0.5): Classifier-free guidance weight
- **max_new_tokens** (int, default 1000): Maximum speech tokens to generate

### Output

- **Waveforms**: Float tensor of shape `(batch_size, audio_samples)` at 24kHz sample rate

### Text Normalization

The model automatically applies text normalization via the `punc_norm()` function, which:
- Capitalizes the first letter
- Normalizes punctuation (replaces uncommon characters like "…" with ", ")
- Converts colons and semicolons to commas
- Replaces em-dashes and en-dashes with hyphens
- Ensures proper sentence ending (adds period if missing)
- Removes multiple spaces

Example:
```python
from transformers.models.chatterbox.modeling_chatterbox import punc_norm

text = "hello world... this is a test: with semicolons; and dashes—like this"
normalized = punc_norm(text)
# Output: "Hello world, this is a test, with semicolons, and dashes-like this."
```

## Limitations

- Requires GPU for real-time performance
- Reference audio quality directly affects output quality
- Reference audio should be 6-10 seconds for best results
- English-only model works best for English text; use multilingual model for other languages
- Text tokenizer must be loaded separately via `load_text_tokenizer()` if not passed as parameter
- Generated speech quality depends on T3 sampling parameters (temperature, top_p, etc.)

## Citation

If you use Chatterbox in your research, please cite:

```bibtex
@misc{chatterbox2025,
  title={Chatterbox: High-Quality Text-to-Speech Synthesis},
  author={Resemble AI},
  year={2025},
  publisher={GitHub},
  url={https://github.com/resemble-ai/chatterbox}
}
```

## ChatterboxConfig

[[autodoc]] ChatterboxConfig
    - english_only
    - multilingual

## T3Config

[[autodoc]] T3Config

## ChatterboxFeatureExtractor

[[autodoc]] ChatterboxFeatureExtractor

## ChatterboxModel

[[autodoc]] ChatterboxModel
    - forward
    - generate
    - prepare_text_tokens
    - prepare_conditionals
    - load_text_tokenizer

