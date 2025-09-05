# ParakeetCTC

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

ParakeetCTC is a complete speech recognition model that combines a [Fast Conformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer) encoder with a CTC (Connectionist Temporal Classification) decoder for automatic speech recognition. It is part of the **Parakeet** family of models from NVIDIA NeMo that use the FastConformer architecture as their encoder foundation.

The ParakeetCTC model consists of two main components:

1. **Fast Conformer Encoder**: A linearly scalable Conformer architecture that processes mel-spectrogram features and reduces sequence length through subsampling. This is more efficient version of the Conformer Encoder found in [FastSpeech2Conformer](./fastspeech2_conformer.md) (see [`ParakeetEncoder`] for the encoder implementation and details).

2. **CTC Decoder**: Simple but effective decoder consisting of:
   - 1D convolution projection from encoder hidden size to vocabulary size (for optimal NeMo compatibility).
   - CTC loss computation for training.
   - Greedy CTC decoding for inference.

ParakeetCTC achieves state-of-the-art accuracy while being computationally efficient, making it suitable for both research and production deployments. The model supports various vocabulary sizes and can handle character-level, subword, or word-level tokenization.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).

## Usage

### Speech Recognition With `pipeline`

Hugging Face's `pipeline` interface can be used to conveniently transcribe audio:
```python
from transformers import pipeline

pipe = pipeline(
    task="automatic-speech-recognition",
    model="bezzam/parakeet-ctc-1.1b-hf",
    device=0
)
res = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print("Transcription:", res["text"])
# Transcription: i have a dream that one day this nation will rise up and live out the true meaning of its creed
```

### With `ParakeetForCTC`

For a more manual approach:
```python
import torch
from transformers import ParakeetForCTC, AutoProcessor
from datasets import load_dataset, Audio

repo_id = "bezzam/parakeet-ctc-1.1b-hf"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load complete CTC model
model = ParakeetForCTC.from_pretrained(repo_id).to(torch_device).eval()
processor = AutoProcessor.from_pretrained(repo_id)

# Load test audio
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

# Convert numpy array to tensor efficiently
audio_array = ds[0]['audio']['array']
raw_audio = torch.from_numpy(audio_array)

# Extract mel-spectrogram features (automatic padding)
inputs = processor(raw_audio).to(torch_device)

# Get CTC outputs
with torch.no_grad():
    outputs = model(
        input_features=inputs.input_features,
        attention_mask=inputs.attention_mask,
    )
ctc_logits = outputs.logits  # Shape: (batch, time, vocab_size)

# Get transcription (greedy decoding)
predicted_ids = model.generate(**inputs)
text = processor.decode(predicted_ids)
print("Transcription:", text)
```

### Using `AutoModel`

You can also use the `AutoModel` API which automatically loads `ParakeetForCTC` for speech recognition tasks:

```python
import torch
from transformers import AutoModelForCTC
from datasets import load_dataset, Audio

repo_id = "bezzam/parakeet-ctc-1.1b-hf"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load complete CTC model
model = AutoModelForCTC.from_pretrained(repo_id).to(torch_device).eval()
```

### Batch Processing

`ParakeetForCTC` efficiently handles batches of audio samples with different lengths when passed as a list. The processor automatically handles padding and length computation:

```python
import torch
from transformers import ParakeetForCTC, AutoProcessor
from datasets import load_dataset, Audio

repo_id = "bezzam/parakeet-ctc-1.1b-hf"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load complete CTC model
model = ParakeetForCTC.from_pretrained(repo_id).to(torch_device).eval()
feature_extractor = AutoProcessor.from_pretrained(repo_id)

# Load test audio samples
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
batch_audio = [torch.from_numpy(ds[i]["audio"]["array"]) for i in range(2)]

# Batch processing
inputs = processor(batch_audio, return_tensors="pt", padding=True).to(torch_device)
predicted_ids = model.generate(**inputs)
texts = processor.batch_decode(predicted_ids)
print("Transcription (batch):", texts)
```

## Model Architecture

`ParakeetForCTC` follows an encoder-decoder architecture specifically designed for CTC-based speech recognition:

### Components

1. **Fast Conformer Encoder**: Processes mel-spectrogram features using:
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

- `ParakeetForCTC` is specifically designed for speech recognition tasks using CTC. For other speech tasks, consider the base [`ParakeetEncoder`] with custom decoders.
- The model expects mel-spectrogram features as input. Use [`ParakeetFeatureExtractor`] for proper preprocessing.
- For best results, ensure your audio is sampled at 16kHz as expected by the feature extractor.
- The `model.generate()` method performs CTC decoding internally and returns already-decoded token sequences. Simply use `tokenizer.decode()` to convert these to text.
- The model automatically handles sequence length computation and attention masking for batched inputs.

## Conversion from NeMo

Parakeet models can be converted from NVIDIA NeMo CTC model checkpoints with full feature parity and numerical equivalence:

```bash
python src/transformers/models/parakeet/convert_nemo_to_hf.py \
    --path_to_nemo_model /path/to/nemo_ctc_model.nemo \
    --output_dir ./parakeet-ctc-hf \
    --verify --push_to_hub hub_user/repo_name
```

## ParakeetConfig

[[autodoc]] ParakeetConfig 

## ParakeetEncoderConfig

[[autodoc]] ParakeetEncoderConfig 

## ParakeetCTCTokenizer

[[autodoc]] ParakeetCTCTokenizer 

## ParakeetFeatureExtractor

[[autodoc]] ParakeetFeatureExtractor
    - __call__

## ParakeetProcessor

[[autodoc]] ParakeetProcessor
    - __call__
    - batch_decode
    - decode

## ParakeetForCTC

[[autodoc]] ParakeetForCTC

