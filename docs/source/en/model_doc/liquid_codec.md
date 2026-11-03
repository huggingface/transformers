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
*This model was contributed to Hugging Face Transformers on 2026-10-23.*


# Liquid Codec

## Overview

Liquid Codec is the speech codec of [LFM3-Audio](lfm3_audio) by [Liquid AI](https://liquid.ai/). It turns 24 kHz audio
into 12 parallel codebooks of 2016 codes at 12.5 frames per second and back: an STFT front end, a causal
[LFM3](lfm3) encoder, finite scalar quantization, and a causal LFM3 decoder followed by an inverse STFT. The decoder is
causal, so audio can be synthesized frame by frame as the codes arrive.

## Usage

The LFM3-Audio and LFM3-Omni processors load the same checkpoint as `processor.audio_tokenizer`.

```python
import torch
from transformers import LiquidCodecModel

codec = LiquidCodecModel.from_pretrained("LiquidAI/Liquid-Codec", dtype=torch.bfloat16, device_map="cuda")

waveform = torch.randn(1, 24_000 * 3, device="cuda")  # 24 kHz
codes = codec.encode(waveform).audio_codes  # (1, 12, 38)
reconstructed = codec.decode(codes).audio_values  # (1, 38 * 1920), float32

# streaming synthesis: every chunk returns the samples that no later frame changes any more
chunks, past_key_values, overlap_cache = [], None, None
for i in range(codes.shape[-1]):
    out = codec.decode(codes[..., i : i + 1], past_key_values, overlap_cache, use_streaming=True, finalize=False)
    chunks.append(out.audio_values)
    past_key_values, overlap_cache = out.decoder_past_key_values, out.overlap_cache
chunks.append(codec.decode(codes[..., :0], past_key_values, overlap_cache, finalize=True).audio_values)
streamed = torch.cat(chunks, dim=-1)
```

## LiquidCodecConfig

[[autodoc]] LiquidCodecConfig

## LiquidCodecOverlapAddCache

[[autodoc]] LiquidCodecOverlapAddCache

## LiquidCodecModel

[[autodoc]] LiquidCodecModel
    - encode
    - decode
    - forward
