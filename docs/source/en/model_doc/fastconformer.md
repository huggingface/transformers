# FastConformer

## Overview

The FastConformer model was proposed in [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084). FastConformer is an optimized version of the Conformer architecture for automatic speech recognition (ASR) that reduces computational complexity while maintaining high accuracy. FastConformer replaces the quadratic self-attention with a linearly scalable attention mechanism while preserving the modeling capability of Conformer. Specifically, it applies 8x downsampling, depth-wise separable convolutions and optionally limits attention context for improved memory efficiency. The proposed FastConformer achieves comparable recognition accuracy with significantly reduced computational costs, making it practical for deployment in resource-constrained environments.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## Model Architecture

FastConformer is primarily used as an encoder architecture that serves as the foundation for the **Parakeet** family of speech recognition models:

- **FastConformer**: Encoder-only model for feature extraction and as a base for custom decoders
- **ParakeetCTC**: Complete speech recognition model with FastConformer encoder + CTC decoder (see [`ParakeetCTC`])
- **Future extensions**: ParakeetTDT (Transducer) and ParakeetRNNT (RNN-Transducer) models

## Usage

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

### Using FastConformer as a Base for Custom Decoders

```python
import torch
from transformers import FastConformerEncoder, FastConformerConfig, FastConformerFeatureExtractor

# Create custom configuration
config = FastConformerConfig(
    hidden_size=512,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_mel_bins=80,
    subsampling_factor=8
)

# Initialize encoder
encoder = FastConformerEncoder(config)
feature_extractor = FastConformerFeatureExtractor()

# Process audio
raw_audio = torch.randn(2, 16000)  # Batch of 2 audio samples
audio_lengths = torch.tensor([16000, 12000])

features = feature_extractor(
    raw_audio,
    audio_lengths=audio_lengths,
    sampling_rate=16000,
    return_tensors="pt"
)

# Get encoder outputs for your custom decoder
encoder_outputs = encoder(
    input_features=features.input_features,
    attention_mask=features.attention_mask,
    input_lengths=features.input_lengths,
    output_hidden_states=True,  # Get all layer outputs
    output_attentions=True      # Get attention weights
)

# Use outputs for your custom decoder
hidden_states = encoder_outputs.last_hidden_state      # Final layer output
all_hidden_states = encoder_outputs.hidden_states      # All layer outputs
attention_weights = encoder_outputs.attentions         # Attention weights

# Example: Simple linear decoder
import torch.nn as nn
vocab_size = 1000
custom_decoder = nn.Linear(config.hidden_size, vocab_size)
logits = custom_decoder(hidden_states)
```

For end-to-end speech recognition, use the ParakeetCTC model which combines FastConformer encoder with a CTC decoder. See the [`ParakeetCTC`] documentation for detailed usage examples.

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
- **Modular Design**: Can be used standalone or as encoder for various decoder types

### Model Variants

- **FastConformerEncoder**: Core encoder implementation
- **FastConformerModel**: Complete encoder model with additional utilities
- **Integration**: Used as encoder in ParakeetCTC, ParakeetTDT and ParakeetRNNT models

## Conversion from NeMo

FastConformer models can be converted from NVIDIA NeMo format. The conversion process extracts the encoder component and creates compatible configurations:

```bash
python src/transformers/models/parakeet_ctc/convert_nemo_to_parakeet_ctc.py \
    --input_path /path/to/nemo_model.nemo \
    --output_dir ./fastconformer-hf
```


## FastConformerConfig

[[autodoc]] FastConformerConfig

## FastConformerFeatureExtractor

[[autodoc]] FastConformerFeatureExtractor

## FastConformerPreTrainedModel

[[autodoc]] FastConformerPreTrainedModel

## FastConformerModel

[[autodoc]] FastConformerModel

## FastConformerEncoder

[[autodoc]] FastConformerEncoder 