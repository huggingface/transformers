# FastConformer

## Overview

The FastConformer model was proposed in [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084). FastConformer is an optimized version of the Conformer architecture for automatic speech recognition (ASR) that reduces computational complexity while maintaining high accuracy. FastConformer, replaces the quadratic self-attention with a linearly scalable attention mechanism while preserving the modeling capability of Conformer. Specifically, it applies 8x downsampling, depth-wise seperable convolutions and optionally limit attention context for improved memory efficiency. The proposed FastConformer achieves comparable recognition accuracy with significantly reduced computational costs, making it practical for deployment in resource-constrained environments.*

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## Usage

FastConformer is primarily designed for automatic speech recognition tasks. The model processes mel-spectrogram features extracted from raw audio and produces contextualized representations suitable for downstream ASR tasks. The implementation includes both encoder-only models for feature extraction and Parakeet CTC models for speech recognition.

### CTC Speech Recognition

For speech recognition tasks, use the complete CTC models that include both the FastConformer encoder and CTC decoder:

```python
import torch
from transformers import AutoModelForCTC, AutoFeatureExtractor

# Load CTC model and feature extractor
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Prepare audio input (example with random data)
# In practice, you would load real audio data
raw_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
audio_lengths = torch.tensor([16000])

# Extract mel-spectrogram features
features = feature_extractor(
    raw_audio, 
    audio_lengths=audio_lengths, 
    sampling_rate=16000,
    return_tensors="pt"
)

# Get CTC outputs
with torch.no_grad():
    outputs = model(
        input_features=features.input_features,
        attention_mask=features.attention_mask,
        input_lengths=features.input_lengths
    )

# CTC logits for each time step
ctc_logits = outputs.logits  # Shape: (batch, time, vocab_size)

# Generate decoded token sequences using CTC decoding
decoded_sequences = model.generate_speech_recognition_outputs(
    input_features=features.input_features,
    attention_mask=features.attention_mask,
    input_lengths=features.input_lengths,
)
print("Decoded tokens:", decoded_sequences[0])
```

### Direct ParakeetCTC Usage

You can also use the ParakeetCTC model directly:

```python
import torch
from transformers import ParakeetCTC, AutoFeatureExtractor

# Load model and feature extractor
model = ParakeetCTC.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Process audio
raw_audio = torch.randn(1, 16000)
audio_lengths = torch.tensor([16000])

features = feature_extractor(
    raw_audio, 
    audio_lengths=audio_lengths, 
    sampling_rate=16000,
    return_tensors="pt"
)

# Forward pass
with torch.no_grad():
    outputs = model(
        input_features=features.input_features,
        attention_mask=features.attention_mask,
        input_lengths=features.input_lengths
    )

# CTC decoding
decoded_sequences = model.generate_speech_recognition_outputs(
    input_features=features.input_features,
    attention_mask=features.attention_mask,
    input_lengths=features.input_lengths,
)
```

### Encoder-Only Usage

For feature extraction or as a base for custom decoders, you can use the encoder directly:

```python
import torch
from transformers import AutoModel, AutoFeatureExtractor

# Load encoder model and feature extractor
model = AutoModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Extract features
raw_audio = torch.randn(1, 16000)
audio_lengths = torch.tensor([16000])

features = feature_extractor(
    raw_audio, 
    audio_lengths=audio_lengths, 
    sampling_rate=16000,
    return_tensors="pt"
)

# Get encoder outputs
with torch.no_grad():
    outputs = model(
        input_features=features.input_features,
        attention_mask=features.attention_mask,
        input_lengths=features.input_lengths
    )

encoder_hidden_states = outputs.last_hidden_state  # Shape: (batch, time, hidden_size)
```

### Processing batches with different lengths

```python
import torch
from transformers import AutoModelForCTC, AutoFeatureExtractor

model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Example with two audio samples of different lengths
audio1 = torch.randn(8000)   # 0.5 seconds
audio2 = torch.randn(12000)  # 0.75 seconds

# Pad to same length
max_length = max(len(audio1), len(audio2))
padded_audio1 = torch.cat([audio1, torch.zeros(max_length - len(audio1))])
padded_audio2 = torch.cat([audio2, torch.zeros(max_length - len(audio2))])

batch_audio = torch.stack([padded_audio1, padded_audio2])
audio_lengths = torch.tensor([len(audio1), len(audio2)])

# Extract features
features = feature_extractor(
    batch_audio,
    audio_lengths=audio_lengths,
    sampling_rate=16000,
    return_tensors="pt"
)

# Process through model
with torch.no_grad():
    outputs = model(
        input_features=features.input_features,
        attention_mask=features.attention_mask,
        input_lengths=features.input_lengths
    )

# Batch CTC decoding
decoded_sequences = model.generate_speech_recognition_outputs(
    input_features=features.input_features,
    attention_mask=features.attention_mask,
    input_lengths=features.input_lengths,
)
print("Batch decoded tokens:", decoded_sequences)
```

## Model Architecture

FastConformer follows the Conformer architecture but with optimizations for efficiency:

1. **Subsampling**: Reduces the input sequence length by a factor (typically 8x) using convolutional layers
2. **Conformer Blocks**: Each block contains:
   - Feed-forward module (1/2 scale)
   - Multi-head self-attention with relative positional encoding  
   - Convolution module with depthwise separable convolutions
   - Feed-forward module (1/2 scale)
   - Layer normalization and residual connections
3. **Relative Positional Encoding**: Uses Transformer-XL style relative positional encodings
4. **Efficient Attention**: Optimized attention mechanisms for reduced computational complexity

### Key Features

- **Linear Scalability**: Attention complexity reduced from O(n²) to O(n)
- **NeMo Compatibility**: Weights can be converted from NVIDIA NeMo models
- **Flexible Configuration**: Supports various model sizes and architectures
- **Batch Processing**: Efficient handling of variable-length sequences
- **CTC Support**: Complete CTC implementation for speech recognition

## Conversion from NeMo

FastConformer models can be converted from NVIDIA NeMo format using the provided conversion script:

```bash
python convert_nemo_fastconformer_to_hf.py \
    --model_name nvidia/parakeet-ctc-1.1b \
    --output_dir ./fastconformer-hf \
    --verify
```

The conversion process:
1. Extracts the FastConformer encoder from the NeMo model
2. Maps weight names to HuggingFace conventions
3. Creates compatible configuration and feature extractor
4. Verifies numerical equivalence between implementations

## FastConformerConfig

[[autodoc]] FastConformerConfig

## ParakeetCTCConfig

[[autodoc]] ParakeetCTCConfig

## FastConformerFeatureExtractor

[[autodoc]] FastConformerFeatureExtractor
    - __call__

## FastConformerTokenizer

[[autodoc]] FastConformerTokenizer
    - decode_ctc_tokens
    - ctc_decode_ids
    - batch_decode

## FastConformerModel

[[autodoc]] FastConformerModel
    - forward

## FastConformerEncoder

[[autodoc]] FastConformerEncoder
    - forward

## ParakeetCTC

[[autodoc]] ParakeetCTC 