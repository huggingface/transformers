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
*This model was released on 2024-10-21 and added to Hugging Face Transformers on 2025-01-10 and contributed by [eustlb](https://huggingface.co/eustlb).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Moonshine

[Moonshine](https://huggingface.co/papers/2410.15608) is a family of speech recognition models designed for live transcription and voice command processing. Utilizing an encoder-decoder transformer architecture, Moonshine incorporates Rotary Position Embedding (RoPE) instead of traditional absolute position embeddings, enabling it to process audio inputs of any length without zero-padding. This approach enhances inference efficiency. Compared to OpenAI's Whisper tiny-en, Moonshine Tiny achieves a 5x reduction in compute requirements for transcribing a 10-second speech segment without increasing word error rates. Key architectural improvements include the use of SwiGLU activation in the decoder layers and the adoption of RoPE for flexible audio handling.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="UsefulSensors/moonshine-base", dtype="auto")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="MoonshineForConditionalGeneration">

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, MoonshineForConditionalGeneration

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation").sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base", dtype="auto")

inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
predicted_ids = model.generate(**inputs)
print(f"Transcription: {processor.batch_decode(predicted_ids)[0]}")
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
