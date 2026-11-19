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

Liquid Codec is the speech codec of [LFM3-Audio](./lfm3_audio) by [Liquid AI](https://liquid.ai/). It turns 24 kHz audio
into 12 parallel codebooks of 2016 codes at 12.5 frames per second and back: a causal [LFM3](./lfm3) encoder over
the spectrogram of [`LiquidCodecFeatureExtractor`], finite scalar quantization, and a causal LFM3 decoder followed by an
inverse STFT. The decoder is causal, so audio can be synthesized frame by frame as the codes arrive.

## Usage

[`LiquidCodecFeatureExtractor`] computes the spectrogram the codec encodes, [`~LiquidCodecModel.encode`] turns it into
audio codes, and [`~LiquidCodecModel.decode`] turns the codes back into audio. The [LFM3-Audio](./lfm3_audio)
processor loads the codec as `processor.audio_tokenizer`.

```python
from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio

model_id, revision = "LiquidAI/Liquid-Codec", "refs/pr/1"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)

url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
audio = load_audio(url, sampling_rate=feature_extractor.sampling_rate)
# spectrogram computed on the model's device, as the original implementation does
inputs = feature_extractor(
    audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", device=model.device
).to(model.device, model.dtype)

audio_codes = model.encode(**inputs).audio_codes
print("Audio codes shape:", audio_codes.shape)
# Audio codes shape: torch.Size([1, 12, 74])
audio_values = model.decode(audio_codes).audio_values
print("Audio values shape:", audio_values.shape)
# Audio values shape: torch.Size([1, 142080])

# Equivalently, you can do encoding and decoding in one step
model_output = model(**inputs)
audio_codes = model_output.audio_codes
audio_values = model_output.audio_values
```

### Batch processing

Clips of different lengths are padded to the longest one. Pass the masks along: `encode` returns the
`audio_codes_mask` of the valid frames of each clip, and `decode` the `audio_values_mask` of its valid samples. The
encoder and the decoder are causal, so each clip is coded exactly as it would be alone.

```python
from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio

model_id, revision = "LiquidAI/Liquid-Codec", "refs/pr/1"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)

samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
audios = [
    load_audio(samples + name, sampling_rate=feature_extractor.sampling_rate)
    for name in ("librispeech_mr_quilter.wav", "dude_where_is_my_car.wav")
]
inputs = feature_extractor(
    audio=audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", device=model.device
).to(model.device, model.dtype)

encoder_output = model.encode(**inputs)
print("Audio codes shape:", encoder_output.audio_codes.shape)
# Audio codes shape: torch.Size([2, 12, 814])
print("Frames per clip:", encoder_output.audio_codes_mask.sum(-1).tolist())
# Frames per clip: [74, 814]

decoder_output = model.decode(encoder_output.audio_codes, encoder_output.audio_codes_mask)
print("Samples per clip:", decoder_output.audio_values_mask.sum(-1).tolist())
# Samples per clip: [142080, 1562880]

# drop the padding of each clip
audio_values = [
    values[mask] for values, mask in zip(decoder_output.audio_values, decoder_output.audio_values_mask)
]
```

### Streaming decoding

The decoder is causal, so audio can be synthesized chunk by chunk as the codes arrive. With `use_streaming=True`,
`decode` returns two caches to pass to the next chunk: `decoder_cache`, the state of the decoder, and
`overlap_add_cache`, the last samples of the chunk, which the next chunk still contributes to. At the end of the
stream, `overlap_add_cache.flush()` returns those last samples.

```python
import torch
from transformers import AutoFeatureExtractor, AutoModel
from transformers.audio_utils import load_audio

model_id, revision = "LiquidAI/Liquid-Codec", "refs/pr/1"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)

url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
audio = load_audio(url, sampling_rate=feature_extractor.sampling_rate)
inputs = feature_extractor(
    audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", device=model.device
).to(model.device, model.dtype)
audio_codes = model.encode(**inputs).audio_codes

chunk_size = 4  # codec frames per chunk
decoder_cache, overlap_add_cache = None, None
chunks = []
for start in range(0, audio_codes.shape[-1], chunk_size):
    outputs = model.decode(
        audio_codes[..., start : start + chunk_size],
        decoder_cache=decoder_cache,
        overlap_add_cache=overlap_add_cache,
        use_streaming=True,
    )
    decoder_cache, overlap_add_cache = outputs.decoder_cache, outputs.overlap_add_cache
    chunks.append(outputs.audio_values)  # play or send this chunk

# end of the stream
chunks.append(overlap_add_cache.flush().audio_values)

audio_values = torch.cat(chunks, dim=-1)
print("Audio values shape:", audio_values.shape)
# Audio values shape: torch.Size([1, 142080])
```

A batch can be streamed too: pass the `audio_codes_mask` of each chunk to `decode`, and every chunk, as well as
`flush()`, returns its `audio_values_mask`.

## LiquidCodecConfig

[[autodoc]] LiquidCodecConfig

## LiquidCodecEncoderConfig

[[autodoc]] LiquidCodecEncoderConfig

## LiquidCodecDecoderConfig

[[autodoc]] LiquidCodecDecoderConfig

## LiquidCodecFeatureExtractor

[[autodoc]] LiquidCodecFeatureExtractor
    - __call__

## LiquidCodecOverlapAddCache

[[autodoc]] LiquidCodecOverlapAddCache
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
