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
*This model was contributed to Hugging Face Transformers on 2026-09-19.*

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

## Usage

Convert the original checkpoint once to save the native model configuration, feature extractor, and multimodal
chat template. The command below writes a local checkpoint:

```bash
python -m transformers.models.lfm2_audio.convert_lfm2_audio_to_hf \
    --checkpoint_path LiquidAI/LFM2.5-Audio-1.5B \
    --output_dir ./LFM2.5-Audio-1.5B-hf
```

The converted checkpoint loads through the standard Auto classes. The examples below use
[`kadirnar/LFM2.5-Audio-1.5B-hf`](https://huggingface.co/kadirnar/LFM2.5-Audio-1.5B-hf).
To use your own conversion, replace the model ID with the local output directory.

### Automatic speech recognition

```python
import torch
from datasets import Audio, load_dataset

from transformers import AutoModelForMultimodalLM, AutoProcessor


model_id = "kadirnar/LFM2.5-Audio-1.5B-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
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

Generation settings follow the standard priority: call arguments, an explicit `GenerationConfig`, then
`model.generation_config`. Supported settings include `max_new_tokens`, `max_length`, `do_sample`, `temperature`,
`top_k`, and `eos_token_id`. Audio and text can use separate sampling settings through the `audio_temperature`,
`audio_top_k`, `text_temperature`, and `text_top_k` fields of `GenerationConfig`; the corresponding call arguments
remain supported. Generation uses one beam and returns an `Lfm2AudioGenerateOutput`. Unsupported generation options
raise an error when set to non-default values.

The processor supports PyTorch tensors (`return_tensors="pt"`), NumPy arrays (`"np"`), and Python lists (`None`).
Text and audio-code forward passes can be compiled with `torch.compile(fullgraph=True)`. Audio-input forward passes
also require `torch._dynamo.config.capture_dynamic_output_shape_ops = True` because the number of unpadded encoder
features depends on the audio lengths.

## Lfm2AudioConfig

[[autodoc]] Lfm2AudioConfig

## Lfm2AudioFeatureExtractor

[[autodoc]] Lfm2AudioFeatureExtractor
    - __call__

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
