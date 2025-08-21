# ParakeetCTC

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

ParakeetCTC is a complete speech recognition model that combines a FastConformer encoder with a CTC (Connectionist Temporal Classification) decoder for automatic speech recognition. It is part of the **Parakeet** family of models from NVIDIA NeMo that use the FastConformer architecture as their encoder foundation.

The ParakeetCTC model consists of two main components:

1. **FastConformer Encoder**: A linearly scalable Conformer architecture that processes mel-spectrogram features and reduces sequence length through subsampling (see [`FastConformerModel`] for encoder details).

2. **CTC Decoder**: Simple but effective decoder consisting of:
   - 1D convolution projection from encoder hidden size to vocabulary size (for optimal NeMo compatibility)
   - CTC loss computation for training
   - Greedy CTC decoding for inference

ParakeetCTC achieves state-of-the-art accuracy while being computationally efficient, making it suitable for both research and production deployments. The model supports various vocabulary sizes and can handle character-level, subword, or word-level tokenization.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## Usage

### Basic Speech Recognition

```python
import torch
from transformers import ParakeetForCTC, AutoFeatureExtractor, AutoTokenizer
from datasets import load_dataset, Audio

# Load complete CTC model
model = ParakeetForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")
tokenizer = AutoTokenizer.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Load test audio
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

# Convert numpy array to tensor efficiently
audio_array = ds[0]['audio']['array']
raw_audio = torch.from_numpy(audio_array)

# Extract mel-spectrogram features (automatic padding)
inputs = feature_extractor(
    raw_audio, 
    sampling_rate=feature_extractor.sampling_rate
)

# Get CTC outputs
with torch.no_grad():
    outputs = model(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask,
    )

# CTC logits for each time step
ctc_logits = outputs.logits  # Shape: (batch, time, vocab_size)

# Generate decoded token sequences using CTC decoding
decoded_sequences = model.generate(
    input_features=inputs.input_features,
    attention_mask=inputs.attention_mask,
)

# Decode to text using CTC-aware tokenizer
text = tokenizer.decode(decoded_sequences[0])
print("Transcription:", text)
```

### Using AutoModel for Speech Recognition

You can also use the `AutoModel` API which automatically loads `ParakeetForCTC` for speech recognition tasks:

```python
import torch
from transformers import AutoModel, AutoFeatureExtractor, AutoTokenizer
from datasets import load_dataset, Audio

# Load using AutoModel (automatically detects `ParakeetForCTC`)
model = AutoModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")
tokenizer = AutoTokenizer.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Load test audio
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

# Convert numpy array to tensor efficiently
audio_array = ds[0]['audio']['array']
raw_audio = torch.from_numpy(audio_array)

# Extract features (automatic length inference and padding)
inputs = feature_extractor(
    raw_audio, 
    sampling_rate=feature_extractor.sampling_rate
)

# Forward pass
with torch.no_grad():
    outputs = model(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask,
    )

# Generate transcription
decoded_sequences = model.generate(
    input_features=inputs.input_features,
    attention_mask=inputs.attention_mask,
)

text = tokenizer.decode(decoded_sequences[0])
print("Transcription:", text)
```

### Batch Processing

`ParakeetForCTC` efficiently handles batches of audio samples with different lengths when passed as a list. The feature extractor automatically handles padding and length computation:

```python
import torch
from transformers import ParakeetForCTC, AutoFeatureExtractor, AutoTokenizer
from datasets import load_dataset, Audio

model = ParakeetForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b")
feature_extractor = AutoFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-1.1b")
tokenizer = AutoTokenizer.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Load test audio samples
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

# Example with two audio samples of different lengths
audio1 = torch.from_numpy(ds[0]['audio']['array'])  # First sample
audio2 = torch.from_numpy(ds[1]['audio']['array'])  # Second sample

batch_audio = [audio1, audio2]  # Simply pass as list for automatic padding

inputs = feature_extractor(
    batch_audio,
    sampling_rate=feature_extractor.sampling_rate
)

# Process through model
with torch.no_grad():
    outputs = model(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask,
    )

# Batch CTC decoding
decoded_sequences = model.generate(
    input_features=inputs.input_features,
    attention_mask=inputs.attention_mask,
)

# Decode batch to text
texts = tokenizer.batch_decode(decoded_sequences)
print("Batch transcriptions:", texts)
```

## Model Architecture

ParakeetCTC follows an encoder-decoder architecture specifically designed for CTC-based speech recognition:

### Components

1. **FastConformer Encoder**: Processes mel-spectrogram features using:
   - Subsampling layer (8x downsampling with convolutional layers)
   - Multiple Conformer blocks with attention and convolution modules
   - Relative positional encoding for better sequence modeling
   - Efficient linear-scalable attention mechanisms

2. **CTC Decoder**: Simple but effective decoder consisting of:
   - 1D convolution projection from encoder hidden size to vocabulary size (for optimal NeMo compatibility)
   - CTC loss computation for training
   - Greedy CTC decoding for inference

3. **Feature Processing**: Mel-spectrogram feature extraction with:
   - NeMo-compatible preprocessing for numerical equivalence
   - Proper attention masking for variable-length sequences
   - Batch processing with length awareness

### Key Features

- **Efficient Architecture**: 8x temporal downsampling reduces computational complexity
- **CTC Decoding**: Handles variable-length input-output alignment without explicit alignment
- **Batch Processing**: Optimized for processing multiple audio samples simultaneously
- **NeMo Compatibility**: Supports conversion from NVIDIA NeMo model checkpoints

## Usage Tips

- ParakeetCTC is specifically designed for speech recognition tasks using CTC. For other speech tasks, consider the base [`FastConformerModel`] with custom decoders.
- The model expects mel-spectrogram features as input. Use [`FastConformerFeatureExtractor`] for proper preprocessing.
- For best results, ensure your audio is sampled at 16kHz as expected by the feature extractor.
- The `model.generate()` method performs CTC decoding internally and returns already-decoded token sequences. Simply use `tokenizer.decode()` to convert these to text.
- The model automatically handles sequence length computation and attention masking for batched inputs.

## Conversion from NeMo

`ParakeetForCTC` models can be converted from NVIDIA NeMo CTC model checkpoints with full feature parity and numerical equivalence:

```bash
python src/transformers/models/parakeet_ctc/convert_nemo_to_parakeet_ctc.py \
    --input_path /path/to/nemo_ctc_model.nemo \
    --output_dir ./parakeet-ctc-hf
```

## ParakeetCTCConfig

[[autodoc]] ParakeetCTCConfig 

## ParakeetCTCTokenizer

[[autodoc]] ParakeetCTCTokenizer 

## ParakeetCTCPreTrainedModel

[[autodoc]] ParakeetCTCPreTrainedModel 

## ParakeetCTC

[[autodoc]] ParakeetCTC
