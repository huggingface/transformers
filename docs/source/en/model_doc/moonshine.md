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

You can find all the original Moonshine checkpoints under the [Useful Sensors](https://huggingface.co/UsefulSensors) organization.

> [!TIP]
> Click on the Moonshine models in the right sidebar for more examples of how to apply Moonshine to different speech recognition tasks.

The example below demonstrates how to transcribe speech into text with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="automatic-speech-recognition",
    model="UsefulSensors/moonshine-base",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

</hfoption>
<hfoption id="AutoModel">

```py
# pip install datasets
import torch
from datasets import load_dataset
from transformers import AutoProcessor, MoonshineForConditionalGeneration

processor = AutoProcessor.from_pretrained(
    "UsefulSensors/moonshine-base",
)
model = MoonshineForConditionalGeneration.from_pretrained(
    "UsefulSensors/moonshine-base",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
).to("cuda")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
audio_sample = ds[0]["audio"]

input_features = processor(
    audio_sample["array"],
    sampling_rate=audio_sample["sampling_rate"],
    return_tensors="pt"
)
input_features = input_features.to("cuda", dtype=torch.float16)

predicted_ids = model.generate(**input_features, cache_implementation="static")
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
transcription[0]
```
</hfoption>
</hfoptions>

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

