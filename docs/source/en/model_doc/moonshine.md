<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
          <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
          <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
          <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Moonshine

[Moonshine](https://huggingface.co/papers/2410.15608) is an encoder-decoder speech recognition model optimized for real-time transcription and recognizing voice command. Instead of using traditional absolute position embeddings, Moonshine uses Rotary Position Embedding (RoPE) to handle speech with varying lengths without using padding. This improves efficiency during inference, making it ideal for resource-constrained devices.


You can find all the Moonshine checkpoints on the [Hub](https://huggingface.co/models?search=moonshine).

> [!TIP]
> Click on the Moonshine models in the right sidebar for more examples of how to apply Moonshine to different speech recognition tasks.

The example below demonstrates how to generate a transcription based on an audio file with [`Pipeline`] or the [`AutoModel`] class.



<hfoptions id="usage">
<hfoption id="Pipeline">

```py
# uncomment to install ffmpeg which is needed to decode the audio file
# !brew install ffmpeg

from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="UsefulSensors/moonshine-base")

result = asr("path_to_audio_file")

#Prints the transcription from the audio file
print(result["text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
# uncomment to install librosa which is used for audio and music anlaysis. It is used to preprocess the data.
# !pip install librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-tiny")

audio_array, sr = librosa.load("pathToFile", sr=16000)
inputs = processor(audio_array, return_tensors="pt", sampling_rate=16000)

generated_ids = model.generate(**inputs, max_new_tokens=256)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Transcription: '{transcription}'")
```
</hfoption>
</hfoptions>

## Notes

- Moonshine improves upon Whisper's architecture:
  1. It uses SwiGLU activation instead of GELU in the decoder layers
  2. Most importantly, it replaces absolute position embeddings with Rotary Position Embeddings (RoPE). This allows Moonshine to handle audio inputs of any length, unlike Whisper which is restricted to fixed 30-second windows.

- A guide for automatic speech recognition can be found [here](../tasks/asr)

## MoonshineConfig

[[autodoc]] MoonshineConfig

## MoonshineModel

[[autodoc]] MoonshineModel
    - forward
    - _mask_input_features

## MoonshineForConditionalGeneration

[[autodoc]] MoonshineForConditionalGeneration
    - forward
    - generate

