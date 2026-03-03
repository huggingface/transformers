<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-10-21 and added to Hugging Face Transformers on 2026-02-03.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
          <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
          <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
          <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Moonshine Streaming

Moonshine Streaming is a streaming variant of the [Moonshine](https://huggingface.co/papers/2410.15608) speech recognition model, optimized for real-time transcription with low latency. Like the original Moonshine, it is an encoder-decoder model that uses Rotary Position Embedding (RoPE) for handling variable-length speech efficiently. The streaming architecture includes sliding window attention in the encoder and a context adapter that enables incremental processing of audio chunks.

Moonshine Streaming is available in three sizes: tiny, small, and medium, offering a trade-off between speed and accuracy. It is particularly well-suited for on-device streaming transcription and voice command applications.

You can find all the original Moonshine Streaming checkpoints under the [Useful Sensors](https://huggingface.co/UsefulSensors) organization.

> [!TIP]
> Moonshine Streaming processes raw audio waveforms directly without requiring mel-spectrogram preprocessing, making it efficient for real-time applications.

The example below demonstrates how to transcribe speech into text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="automatic-speech-recognition",
    model="UsefulSensors/moonshine-streaming-tiny",
    dtype=torch.float16,
    device=0
)
pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-tiny")
model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    "UsefulSensors/moonshine-streaming-tiny",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = ds[0]["audio"]

inputs = processor(audio_sample["array"], return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=100)
transcription = processor.decode(generated_ids[0], skip_special_tokens=True)
transcription
```

</hfoption>
</hfoptions>

## MoonshineStreamingProcessor

[[autodoc]] MoonshineStreamingProcessor

## MoonshineStreamingEncoderConfig

[[autodoc]] MoonshineStreamingEncoderConfig

## MoonshineStreamingConfig

[[autodoc]] MoonshineStreamingConfig

## MoonshineStreamingModel

[[autodoc]] MoonshineStreamingModel
    - forward

## MoonshineStreamingForConditionalGeneration

[[autodoc]] MoonshineStreamingForConditionalGeneration
    - forward
    - generate