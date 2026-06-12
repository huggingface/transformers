<!--Copyright 2026 OpenMOSS and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-05.*

# MOSS Audio Tokenizer

[MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer) is the neural audio codec used by
MOSS-TTS. It encodes waveforms into discrete audio codebook tokens and decodes those tokens back into waveform audio.

## Usage

```python
import torch

from transformers import MossAudioTokenizerModel


model_id = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
model = MossAudioTokenizerModel.from_pretrained(model_id, device_map="auto")

audio = torch.randn(1, 1, 24000, device=model.device)

encoded = model.encode(audio, return_dict=True)
audio_codes = encoded.audio_codes

decoded = model.decode(audio_codes, return_dict=True)
audio_values = decoded.audio
```

## MossAudioTokenizerConfig

[[autodoc]] MossAudioTokenizerConfig

## MossAudioTokenizerEncoderConfig

[[autodoc]] MossAudioTokenizerEncoderConfig

## MossAudioTokenizerDecoderConfig

[[autodoc]] MossAudioTokenizerDecoderConfig

## MossAudioTokenizerQuantizerConfig

[[autodoc]] MossAudioTokenizerQuantizerConfig

## MossAudioTokenizerModel

[[autodoc]] MossAudioTokenizerModel
    - encode
    - decode
    - batch_encode
    - batch_decode
    - forward
