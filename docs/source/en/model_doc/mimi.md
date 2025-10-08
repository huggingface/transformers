<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2024-09-18 and contributed by [ylacombe](https://huggingface.co/ylacombe).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Mimi

[Moshi](https://huggingface.co/papers/2306.01147) is a speech-text foundation model designed for real-time dialogue. It addresses the limitations of traditional spoken dialogue systems by integrating speech-to-speech generation, eliminating the need for separate components like voice activity detection, speech recognition, and text-to-speech. Moshi uses a text language model backbone to generate speech as tokens from a neural audio codec's residual quantizer, modeling both its own speech and the user's speech in parallel streams. This approach allows for the handling of overlapping speech and interruptions without explicit speaker turns. Additionally, Moshi introduces a method called "Inner Monologue" to predict time-aligned text tokens before audio tokens, enhancing linguistic quality and enabling streaming speech recognition and text-to-speech. The model operates with a theoretical latency of 160ms and a practical latency of 200ms, making it the first real-time full-duplex spoken large language model. 

<hfoptions id="usage">
<hfoption id="MimiModel">

```py
import torch
from datasets import load_dataset, Audio
from transformers import MimiModel, AutoFeatureExtractor

model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio_sample = librispeech_dummy[-1]["audio"]["array"]
inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(encoder_outputs.audio_codes, inputs["padding_mask"])[0]
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

</hfoption>
</hfoptions>

## MimiConfig

[[autodoc]] MimiConfig

## MimiModel

[[autodoc]] MimiModel
    - decode
    - encode
    - forward

