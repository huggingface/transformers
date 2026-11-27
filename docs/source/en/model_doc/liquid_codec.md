<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-19.*


# Liquid Codec

## Overview

Liquid Codec is the speech codec of LFM3-Audio by [Liquid AI](https://liquid.ai/): 24 kHz audio to 12 codebooks of
2016 codes at 12.5 Hz, and back. Causal [LFM3](./lfm3) encoder and decoder, finite scalar quantization, ISTFT head.

## Usage

```python
from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio

model_id, revision = "LiquidAI/Liquid-Codec", "refs/pr/1"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)

url = "https://hf.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
audio = load_audio(url, sampling_rate=feature_extractor.sampling_rate)
inputs = feature_extractor(
    audio=audio,
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt",
).to(model.device, model.dtype)

audio_codes = model.encode(**inputs).audio_codes
print("Audio codes shape:", audio_codes.shape)
# Audio codes shape: torch.Size([1, 12, 74])
audio_values = model.decode(audio_codes).audio_values
print("Audio values shape:", audio_values.shape)
# Audio values shape: torch.Size([1, 142080])

# or encode and decode in one step
model_output = model(**inputs)
audio_codes = model_output.audio_codes
audio_values = model_output.audio_values
```

### Batch processing

Pass the masks along: `encode` returns `audio_codes_mask`, `decode` returns `audio_values_mask`. Right padding only.

```python
from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio

model_id, revision = "LiquidAI/Liquid-Codec", "refs/pr/1"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)

samples = "https://hf.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
audios = [
    load_audio(samples + name, sampling_rate=feature_extractor.sampling_rate)
    for name in ("librispeech_mr_quilter.wav", "dude_where_is_my_car.wav")
]
inputs = feature_extractor(
    audio=audios,
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt",
).to(model.device, model.dtype)

encoder_output = model.encode(**inputs)
print("Audio codes shape:", encoder_output.audio_codes.shape)
# Audio codes shape: torch.Size([2, 12, 814])
print("Frames per clip:", encoder_output.audio_codes_mask.sum(-1).tolist())
# Frames per clip: [74, 814]

decoder_output = model.decode(
    encoder_output.audio_codes,
    encoder_output.audio_codes_mask,
)
print("Samples per clip:", decoder_output.audio_values_mask.sum(-1).tolist())
# Samples per clip: [142080, 1562880]

# drop the padding
audio_values = [
    values[mask]
    for values, mask in zip(decoder_output.audio_values, decoder_output.audio_values_mask)
]
```

### Streaming decoding

With `use_streaming=True`, `decode` returns `decoder_cache` and `istft_cache` to pass to the next chunk.
`istft_cache.flush()` returns the last samples. `audio_codes_mask` is not supported.

```python
import torch
from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio

model_id, revision = "LiquidAI/Liquid-Codec", "refs/pr/1"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)

url = "https://hf.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
audio = load_audio(url, sampling_rate=feature_extractor.sampling_rate)
inputs = feature_extractor(
    audio=audio,
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt",
).to(model.device, model.dtype)
audio_codes = model.encode(**inputs).audio_codes

chunk_size = 4  # codec frames per chunk, arbitrary
decoder_cache, istft_cache = None, None
chunks = []
for start in range(0, audio_codes.shape[-1], chunk_size):
    outputs = model.decode(
        audio_codes[..., start : start + chunk_size],
        decoder_cache=decoder_cache,
        istft_cache=istft_cache,
        use_streaming=True,
    )
    decoder_cache, istft_cache = outputs.decoder_cache, outputs.istft_cache
    chunks.append(outputs.audio_values)

chunks.append(istft_cache.flush().audio_values)

audio_values = torch.cat(chunks, dim=-1)
print("Audio values shape:", audio_values.shape)
# Audio values shape: torch.Size([1, 142080])
```

## LiquidCodecConfig

[[autodoc]] LiquidCodecConfig

## LiquidCodecEncoderConfig

[[autodoc]] LiquidCodecEncoderConfig

## LiquidCodecDecoderConfig

[[autodoc]] LiquidCodecDecoderConfig

## LiquidCodecFeatureExtractor

[[autodoc]] LiquidCodecFeatureExtractor
    - __call__

## LiquidCodecISTFTCache

[[autodoc]] LiquidCodecISTFTCache
    - update
    - flush

## LiquidCodecModel

[[autodoc]] LiquidCodecModel
    - encode
    - decode
    - forward

## LiquidCodecEncoderModel

[[autodoc]] LiquidCodecEncoderModel
    - forward

## LiquidCodecDecoderModel

[[autodoc]] LiquidCodecDecoderModel
    - forward
