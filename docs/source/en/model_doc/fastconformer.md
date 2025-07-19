# FastConformer

## Overview

The FastConformer model was proposed in [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084). FastConformer is an optimized version of the Conformer architecture for automatic speech recognition (ASR) that reduces computational complexity while maintaining high accuracy. FastConformer replaces the quadratic self-attention with a linearly scalable attention mechanism while preserving the modeling capability of Conformer. Specifically, it applies 8x downsampling, depth-wise separable convolutions and optionally limits attention context for improved memory efficiency. The proposed FastConformer achieves comparable recognition accuracy with significantly reduced computational costs, making it practical for deployment in resource-constrained environments.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## Model Architecture

FastConformer is primarily used as an encoder in the **Parakeet** family of models:

- **FastConformer**: Encoder-only model for feature extraction and as a base for custom decoders
- **ParakeetCTC**: Complete speech recognition model with FastConformer encoder + CTC decoder
- **Future extensions**: ParakeetTDT (Transducer) and ParakeetRNNT (RNN-Transducer) models

## Usage

### CTC Speech Recognition with ParakeetCTC

For complete speech recognition tasks, use ParakeetCTC models that include both the FastConformer encoder and CTC decoder:

```python
import torch
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
from datasets import load_dataset, Audio

# Load complete CTC model
model = AutoModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")
tokenizer = AutoTokenizer.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Load test audio
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

raw_audio = torch.tensor([ds[0]['audio']['array']])
audio_lengths = torch.tensor([len(raw_audio[0])])

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

# Decode to text using CTC-aware tokenizer
text = tokenizer.decode(decoded_sequences[0], ctc_decode=True)
print("Transcription:", text)
```

### Direct ParakeetCTC Usage

You can also import and use the ParakeetCTC model directly:

```python
import torch
from transformers import ParakeetCTC, FastConformerFeatureExtractor, ParakeetCTCTokenizer
from datasets import load_dataset, Audio

# Load model components directly
model = ParakeetCTC.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = FastConformerFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")
tokenizer = ParakeetCTCTokenizer.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Process audio
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

raw_audio = torch.tensor([ds[0]['audio']['array']])
audio_lengths = torch.tensor([len(raw_audio[0])])

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

# Decode to text
text = tokenizer.decode(decoded_sequences[0], ctc_decode=True)
print("Transcription:", text)
```

### Encoder-Only Usage (FastConformer)

For feature extraction or as a base for custom decoders, you can use the FastConformer encoder directly:

```python
import torch
from transformers import FastConformerModel, FastConformerFeatureExtractor

# Load encoder-only model
encoder = FastConformerModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = FastConformerFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Note: When loading from ParakeetCTC checkpoint, you get only the encoder part
raw_audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
audio_lengths = torch.tensor([16000])

features = feature_extractor(
    raw_audio, 
    audio_lengths=audio_lengths, 
    sampling_rate=16000,
    return_tensors="pt"
)

# Get encoder outputs
with torch.no_grad():
    outputs = encoder(
        input_features=features.input_features,
        attention_mask=features.attention_mask,
        input_lengths=features.input_lengths
    )

encoder_hidden_states = outputs.last_hidden_state  # Shape: (batch, subsampled_time, hidden_size)
print(f"Input shape: {features.input_features.shape}")  # (batch, time, mel_bins)
print(f"Output shape: {encoder_hidden_states.shape}")   # (batch, time//8, hidden_size)
```

### Processing Batches with Different Lengths

```python
import torch
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer

model = AutoModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")
tokenizer = AutoTokenizer.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Example with two audio samples of different lengths
audio1 = torch.randn(8000)   # 0.5 seconds at 16kHz
audio2 = torch.randn(12000)  # 0.75 seconds at 16kHz

# Pad to same length for batching
max_length = max(len(audio1), len(audio2))
padded_audio1 = torch.cat([audio1, torch.zeros(max_length - len(audio1))])
padded_audio2 = torch.cat([audio2, torch.zeros(max_length - len(audio2))])

batch_audio = torch.stack([padded_audio1, padded_audio2])
audio_lengths = torch.tensor([len(audio1), len(audio2)])

# Extract features with proper length handling
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

# Decode batch to text
texts = tokenizer.batch_decode(decoded_sequences, ctc_decode=True)
print("Batch transcriptions:", texts)
```

## Model Architecture Details

FastConformer follows the Conformer architecture but with optimizations for efficiency:

1. **Subsampling Layer**: Reduces the input sequence length by a factor of 8 using strided convolutional layers
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
- **CTC Support**: Complete CTC implementation for speech recognition in ParakeetCTC

### Model Variants

- **FastConformerModel**: Encoder-only model for feature extraction
- **ParakeetCTC**: FastConformer encoder + CTC decoder for speech recognition
- **Future**: ParakeetTDT and ParakeetRNNT models will extend this architecture

## Conversion from NeMo

FastConformer and ParakeetCTC models can be converted from NVIDIA NeMo format using the provided conversion script:

```bash
python src/transformers/models/parakeet_ctc/convert_nemo_to_parakeet_ctc.py \
    --input_path /path/to/nemo_model.nemo \
    --output_dir ./parakeet-ctc-hf
```

The conversion process:
1. Extracts the FastConformer encoder and CTC decoder from the NeMo model
2. Maps weight names to HuggingFace conventions using regex patterns
3. Creates compatible configuration, feature extractor, and tokenizer
4. Verifies numerical equivalence between implementations

### Conversion Features

- **Automatic model type detection**: CTC, Transducer, etc.
- **Weight mapping**: Handles NeMo → HF parameter name conversions
- **Config generation**: Creates appropriate FastConformerConfig/ParakeetCTCConfig
- **Tokenizer creation**: Generates ParakeetCTCTokenizer with CTC decoding support
- **Verification**: Tests numerical equivalence with original NeMo model

## FastConformerConfig

[[autodoc]] FastConformerConfig

## ParakeetCTCConfig  

[[autodoc]] ParakeetCTCConfig

## FastConformerFeatureExtractor

[[autodoc]] FastConformerFeatureExtractor
    - __call__

## ParakeetCTCTokenizer

[[autodoc]] ParakeetCTCTokenizer
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
    - forward
    - generate_speech_recognition_outputs 