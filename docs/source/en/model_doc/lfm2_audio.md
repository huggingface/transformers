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
*This model was contributed to Hugging Face Transformers on 2026-08-24.*

# LFM2-Audio

## Overview

[LFM2-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) is an any-to-any speech and text model from
[Liquid AI](https://www.liquid.ai/). It supports automatic speech recognition (ASR), text-to-speech (TTS), and
interleaved speech-to-speech generation.

The model contains four main components:

1. a log-mel frontend and a 17-layer FastConformer audio encoder,
2. a small adapter that projects encoded speech to the LFM2 hidden size,
3. an LFM2 backbone that processes text, input-audio, and output-audio positions in one timeline, and
4. a six-layer depth transformer that predicts eight Mimi codebook tokens for each generated audio frame.

The implementation reuses the native Transformers [`ParakeetEncoderModel`] for the FastConformer. It does not depend
on Moshi or duplicate its source code in an LFM2-Audio-specific folder. For waveform decoding,
`LiquidAI/LFM2.5-Audio-1.5B` includes a compact LFM detokenizer that is loaded lazily by
[`Lfm2AudioProcessor.decode_audio`]. A native Transformers [`MimiModel`] remains available as a fallback for older
checkpoints.

## Usage

The original `LiquidAI/LFM2.5-Audio-1.5B` checkpoint predates its native Transformers integration. Use the concrete
classes below to load that checkpoint directly.

### Automatic speech recognition

```python
import torch
from datasets import Audio, load_dataset

from transformers import Lfm2AudioForConditionalGeneration, Lfm2AudioProcessor


model_id = "LiquidAI/LFM2.5-Audio-1.5B"
processor = Lfm2AudioProcessor.from_pretrained(model_id)
model = Lfm2AudioForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
waveform = dataset[0]["audio"]["array"]

inputs = processor.apply_transcription_request(
    waveform,
    device=model.device,
).to(device=model.device, dtype=model.dtype)
output = model.generate(**inputs, max_new_tokens=256, text_top_k=1)
transcript = processor.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
```

Passing `device=model.device` runs the log-mel frontend on the same accelerator as the model. The frontend remains in
float32 for numerical stability; casting the returned inputs to `model.dtype` afterwards matches Liquid Audio's dtype
boundary and reduces the audio-feature memory by half for a bfloat16 model. Integer token IDs and masks keep their
original dtypes.

### Text-to-speech

Generated audio is represented by eight codebooks. Decode those codes with the detokenizer bundled in the checkpoint.

```python
inputs = processor.apply_text_to_speech_request(
    "The past is just a story we tell ourselves.",
    prompt="Perform TTS. Use the UK male voice.",
).to(device=model.device, dtype=model.dtype)

output = model.generate(
    **inputs,
    max_new_tokens=512,
    audio_temperature=0.8,
    audio_top_k=64,
)
waveform = processor.decode_audio(output.audio_codes)
sampling_rate = processor.output_sampling_rate  # 24 kHz
```

[`Lfm2AudioForConditionalGeneration.generate`] currently supports a batch size of one. `max_new_tokens` counts both
text tokens and audio frames. Use `generation_mode="interleaved"` for speech-to-speech responses containing alternating
text and audio spans.

## Lfm2AudioConfig

[[autodoc]] Lfm2AudioConfig

## Lfm2AudioPreprocessorConfig

[[autodoc]] Lfm2AudioPreprocessorConfig

## Lfm2AudioEncoderConfig

[[autodoc]] Lfm2AudioEncoderConfig

## Lfm2AudioDepthConfig

[[autodoc]] Lfm2AudioDepthConfig

## Lfm2AudioProcessor

[[autodoc]] Lfm2AudioProcessor
    - __call__
    - apply_transcription_request
    - apply_text_to_speech_request
    - decode_audio

## Lfm2AudioModel

[[autodoc]] Lfm2AudioModel
    - forward
    - get_audio_features

## Lfm2AudioDetokenizer

[[autodoc]] Lfm2AudioDetokenizer
    - forward

## Lfm2AudioForConditionalGeneration

[[autodoc]] Lfm2AudioForConditionalGeneration
    - forward
    - generate

## Lfm2AudioModelOutputWithPast

[[autodoc]] Lfm2AudioModelOutputWithPast

## Lfm2AudioConditionalGenerationOutput

[[autodoc]] Lfm2AudioConditionalGenerationOutput

## Lfm2AudioGenerateOutput

[[autodoc]] Lfm2AudioGenerateOutput
