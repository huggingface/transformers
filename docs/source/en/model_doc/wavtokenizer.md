<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2024-08-29 and contributed to Hugging Face Transformers on 2026-09-29.*

# WavTokenizer

## Overview

[WavTokenizer](https://huggingface.co/papers/2408.16532) encodes 24 kHz mono audio into discrete tokens using a
single codebook, at 40 or 75 tokens per second depending on the checkpoint. It can also reconstruct audio
from those tokens.

This implementation supports inference with [`WavTokenizerModel`] for encoding and decoding, or
[`WavTokenizerEncoderModel`] for encoding only. The 40-token-per-second variant supplies the audio tokenizer
for [Apertus 1.5](./apertus1p5). Training the codec is not supported.

This model was contributed by the [Swiss AI Initiative](https://huggingface.co/swiss-ai).
The original implementation is available on [GitHub](https://github.com/jishengpeng/WavTokenizer).

## Available checkpoints

These checkpoints are ready to load with `from_pretrained`:

| Checkpoint | Domain | Tokens per second |
|---|---|---:|
| [Small, 40 tokens/s](https://huggingface.co/swiss-ai/wavtokenizer-small-speech-40token) | Speech | 40 |
| [Small, 75 tokens/s](https://huggingface.co/swiss-ai/wavtokenizer-small-speech-75token) | Speech | 75 |
| [Medium](https://huggingface.co/swiss-ai/wavtokenizer-medium-speech-75token) | Speech | 75 |
| [Medium v2](https://huggingface.co/swiss-ai/wavtokenizer-medium-speech-75token-v2) | Speech | 75 |
| [Medium](https://huggingface.co/swiss-ai/wavtokenizer-medium-music-audio-75token) | Music/audio | 75 |
| [Medium v2](https://huggingface.co/swiss-ai/wavtokenizer-medium-music-audio-75token-v2) | Music/audio | 75 |
| [Large unified](https://huggingface.co/swiss-ai/wavtokenizer-large-unify-40token) | Unified | 40 |
| [Large v2](https://huggingface.co/swiss-ai/wavtokenizer-large-speech-75token-v2) | Speech | 75 |

## Usage example

Load a speech clip, resample it to the checkpoint's sampling rate, and encode and reconstruct it:

```python
import torch
from transformers import AutoFeatureExtractor, WavTokenizerModel
from transformers.audio_utils import load_audio

model_id = "swiss-ai/wavtokenizer-large-unify-40token"
model = WavTokenizerModel.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/belinda.wav",
    sampling_rate=feature_extractor.sampling_rate,
)
inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
with torch.no_grad():
    audio_codes = model.encode(**inputs).audio_codes
    reconstruction = model.decode(audio_codes).audio_values[..., : len(audio)]
```

## Usage notes

- **Audio inputs:** The feature extractor expects mono audio at its configured sampling rate; it does not
  resample or downmix. The example uses `load_audio` to handle both. Decoding can add trailing samples due to
  internal padding, so the example trims the reconstruction to the original length.
- **Batching:** Padding clips of different lengths can change their codes. Encode clips separately when
  you need the same codes as processing each clip on its own.
- **Precision:** Load and run the tokenizer in `float32` (the default). Lower precision can change code
  assignments.

## WavTokenizerConfig

[[autodoc]] WavTokenizerConfig

## WavTokenizerFeatureExtractor

[[autodoc]] WavTokenizerFeatureExtractor
    - __call__

## WavTokenizerEncoderModel

[[autodoc]] WavTokenizerEncoderModel
    - encode
    - forward

## WavTokenizerModel

[[autodoc]] WavTokenizerModel
    - decode
    - encode
    - forward
