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
*This model was contributed to Hugging Face Transformers on 2026-08-04.*

# VoxCPM2

## Overview

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is a 2B-parameter multilingual text-to-speech model from OpenBMB. It accepts text, a reference voice, or a transcribed audio prompt and produces 48 kHz speech. Voice descriptions can be written directly at the start of the text, for example `"(warm, calm voice) Welcome home."`.

VoxCPM2 represents speech with continuous AudioVAE V2 latents instead of discrete audio tokens. A MiniCPM4-based text-semantic language model predicts speech patches autoregressively, while a local encoder, residual language model, and diffusion Transformer model the detail inside each patch. The AudioVAE accepts 16 kHz conditioning audio and decodes the generated latents at 48 kHz.

The original checkpoint stores the main model and AudioVAE weights separately. Download the original repository and convert it to the Transformers layout before loading it natively:

```bash
hf download openbmb/VoxCPM2 --local-dir VoxCPM2-original
python src/transformers/models/voxcpm2/convert_voxcpm2_weights_to_hf.py \
    --input_path VoxCPM2-original \
    --output_path VoxCPM2-transformers
```

## Usage

### Text-to-speech and voice design

Pass text alone for zero-shot generation. Put a voice description in parentheses at the beginning of the text to guide the voice without a reference recording.

```python
import soundfile as sf

from transformers import AutoModelForTextToWaveform, AutoProcessor


checkpoint = "VoxCPM2-transformers"
processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForTextToWaveform.from_pretrained(checkpoint, dtype="auto", device_map="auto")

inputs = processor(
    text="(A warm, calm voice) Welcome to the VoxCPM2 demonstration.",
    return_tensors="pt",
).to(model.device)
audio = model.generate(**inputs)
sf.write("voxcpm2.wav", audio[0].float().cpu().numpy(), model.config.sample_rate)
```

VoxCPM2 generation currently supports one sample at a time.

### Voice cloning

Use `reference_audio` to clone the voice of a short recording. The waveform must be mono and sampled at 16 kHz.

```python
import librosa
import soundfile as sf


reference_audio, sampling_rate = librosa.load("reference.wav", sr=16000, mono=True)
inputs = processor(
    text="This sentence uses the voice from the reference recording.",
    reference_audio=reference_audio,
    sampling_rate=sampling_rate,
    return_tensors="pt",
).to(model.device)
audio = model.generate(**inputs)
sf.write("voxcpm2_clone.wav", audio[0].float().cpu().numpy(), model.config.sample_rate)
```

### Audio continuation

For continuation-based cloning, pass the prompt waveform as `audio` and its exact transcript as `prompt_text`. The same recording can also be supplied as `reference_audio` for additional voice conditioning.

```python
inputs = processor(
    text="The generated speech continues from this point.",
    audio=reference_audio,
    prompt_text="This is the exact transcript of the reference recording.",
    reference_audio=reference_audio,
    sampling_rate=16000,
    return_tensors="pt",
).to(model.device)
audio = model.generate(**inputs)
sf.write("voxcpm2_continuation.wav", audio[0].float().cpu().numpy(), model.config.sample_rate)
```

### Streaming

[`VoxCPM2Model.generate_streaming`] yields waveform chunks as audio patches are generated.

```python
import torch


inputs = processor(text="This sentence is generated as a stream.", return_tensors="pt").to(model.device)
chunks = list(model.generate_streaming(**inputs))
audio = torch.cat(chunks, dim=-1)
sf.write("voxcpm2_streaming.wav", audio[0].float().cpu().numpy(), model.config.sample_rate)
```

## VoxCPM2Config

[[autodoc]] VoxCPM2Config

## VoxCPM2TextConfig

[[autodoc]] VoxCPM2TextConfig

## VoxCPM2EncoderConfig

[[autodoc]] VoxCPM2EncoderConfig

## VoxCPM2DiTConfig

[[autodoc]] VoxCPM2DiTConfig

## VoxCPM2CfmConfig

[[autodoc]] VoxCPM2CfmConfig

## VoxCPM2AudioVAEConfig

[[autodoc]] VoxCPM2AudioVAEConfig

## VoxCPM2Tokenizer

[[autodoc]] VoxCPM2Tokenizer
    - __call__

## VoxCPM2Processor

[[autodoc]] VoxCPM2Processor
    - __call__

## VoxCPM2ModelOutput

[[autodoc]] VoxCPM2ModelOutput

## VoxCPM2GenerationOutput

[[autodoc]] VoxCPM2GenerationOutput

## VoxCPM2Model

[[autodoc]] VoxCPM2Model
    - forward
    - generate
    - generate_streaming
