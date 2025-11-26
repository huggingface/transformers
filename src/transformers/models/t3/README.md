# T3 (Token-To-Token) Model Implementation

## Overview

T3 is a Text-to-Speech (TTS) model that generates speech tokens from text tokens using a LLaMA transformer backbone. The speech tokens can then be decoded by S3Gen to produce mel spectrograms and finally waveforms via HiFTNet.

## Architecture

### Core Components

1. **Backbone**: LLaMA (520M parameters)
   - 30 hidden layers
   - 16 attention heads
   - 1024 hidden size
   - 4096 intermediate size

2. **Embeddings**:
   - Text token embeddings (704 for English, 2454 for multilingual)
   - Speech token embeddings (8194 vocab size)
   - Learned positional embeddings for both text and speech

3. **Conditioning**:
   - Voice Encoder: Extracts speaker embeddings from reference audio
   - Perceiver Resampler: Downsamples conditioning prompts
   - Emotion/Exaggeration control

4. **Output Heads**:
   - Text head: Projects to text vocabulary
   - Speech head: Projects to speech vocabulary

### Model Flow

```
Text Tokens + Speaker Embedding + Emotion → T3 → Speech Tokens → S3Gen → Mel → HiFTNet → Audio
```

## Files Created

### Core Implementation
- `configuration_t3.py`: T3Config class with English-only and multilingual configurations
- `modeling_t3.py`: Main T3Model implementation including:
  - LearnedPositionEmbeddings
  - Perceiver resampler
  - AttentionQKV and AttentionBlock
  - T3CondEnc (conditioning encoder)
  - T3Cond (conditioning dataclass)
  - VoiceEncoder (speaker embedding extraction)
  - AlignmentStreamAnalyzer (multilingual hallucination detection)
- `__init__.py`: Module exports

### Testing
- `tests/models/t3/test_modeling_t3.py`: Comprehensive test suite including:
  - Model initialization tests
  - Forward pass tests
  - Loss computation tests
  - Inference tests
  - Voice encoder tests
  - Save/load tests
  - Configuration tests
- `tests/models/t3/test_t3_pipeline_integration.py`: End-to-end pipeline test

### Utilities
- `convert_t3_checkpoint.py`: Convert chatterbox weights to transformers format

### Registration
- Added T3 to `models/__init__.py`
- Added T3Config to `models/auto/configuration_auto.py`

## Configuration

### English-Only Configuration
```python
from transformers import T3Config

config = T3Config.english_only()
# text_tokens_dict_size = 704
# use_alignment_analyzer = False
```

### Multilingual Configuration
```python
config = T3Config.multilingual()
# text_tokens_dict_size = 2454
# use_alignment_analyzer = True (automatic hallucination detection)
```

## Usage

### Basic Inference

```python
import torch
import numpy as np
from transformers import T3Model, T3Config
from transformers.models.t3.modeling_t3 import T3Cond

# Load model
config = T3Config.english_only()
model = T3Model(config)
model.eval()

# Prepare text tokens (from a text tokenizer)
text_tokens = torch.tensor([[255, 10, 20, 30, 0]])  # [start_token, tokens..., stop_token]

# Extract speaker embedding from reference audio
reference_wav = np.random.randn(32000).astype(np.float32)  # 2 sec at 16kHz
speaker_embeds = model.voice_encoder.embeds_from_wavs([reference_wav], sample_rate=16000)
speaker_emb = torch.from_numpy(speaker_embeds)

# Create conditioning
emotion_adv = torch.ones(1, 1, 1) * 0.5  # neutral
t3_cond = T3Cond(speaker_emb=speaker_emb, emotion_adv=emotion_adv)

# Generate speech tokens
with torch.no_grad():
    speech_tokens = model.inference(
        t3_cond=t3_cond,
        text_tokens=text_tokens[0],
        max_new_tokens=1000,
        temperature=0.8,
        top_p=0.95,
        cfg_weight=0.5,
    )

# speech_tokens can now be passed to S3Gen for mel generation
```

### Converting Chatterbox Weights

The conversion script can merge both T3 and voice encoder weights:

```bash
python src/transformers/models/t3/convert_t3_checkpoint.py \
    --chatterbox_checkpoint_path /path/to/chatterbox/t3_cfg.safetensors \
    --voice_encoder_checkpoint_path /path/to/chatterbox/ve.safetensors \
    --output_path ./t3_hf \
    --config_type english_only \
    --push_to_hub \
    --model_name ResembleAI/t3_cfg
```

If you have the weights in the default chatterbox location:
```bash
python src/transformers/models/t3/convert_t3_checkpoint.py \
    --chatterbox_checkpoint_path /mnt/persistent3/manmay/transformerjs/chatterbox/t3_cfg.safetensors \
    --voice_encoder_checkpoint_path /mnt/persistent3/manmay/transformerjs/ve.safetensors \
    --output_path ./t3_hf \
    --config_type english_only
```

## Pipeline Integration

The T3 model is part of the complete Chatterbox TTS pipeline:

1. **Text Tokenization**: Text → Text Tokens (EnTokenizer)
2. **T3**: Text Tokens → Speech Tokens (this model)
3. **S3Gen**: Speech Tokens → Mel Spectrogram
4. **HiFTNet**: Mel Spectrogram → Waveform

## Special Features

### Voice Cloning
The VoiceEncoder extracts speaker embeddings from reference audio, enabling voice cloning. The speaker embedding is used to condition the generation.

### Emotion Control
The `emotion_adv` parameter (0.0 to 1.0) controls the expressiveness/exaggeration of the generated speech.

### Multilingual Support
The multilingual configuration includes:
- Larger text vocabulary (2454 tokens)
- AlignmentStreamAnalyzer for detecting and preventing hallucinations
- Automatic early stopping detection

### CFG (Classifier-Free Guidance)
The model supports CFG during inference to improve quality by balancing conditional and unconditional generation.

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Backbone | LLaMA 520M |
| Text Vocab (English) | 704 |
| Text Vocab (Multilingual) | 2454 |
| Speech Vocab | 8194 (6561 tokens + specials) |
| Hidden Size | 1024 |
| Num Layers | 30 |
| Attention Heads | 16 |
| Speaker Embed Dim | 256 |
| Max Text Tokens | 2048 |
| Max Speech Tokens | 4096 |

## References

- Original Chatterbox implementation: [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)
- LLaMA: [Meta AI LLaMA](https://ai.meta.com/llama/)
- Perceiver: [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)

