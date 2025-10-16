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
*This model was released on 2024-07-15 and added to Hugging Face Transformers on 2024-08-08.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Qwen2Audio

[Qwen2-Audio](https://huggingface.co/papers/2407.10759) is a large-scale audio-language model that processes various audio inputs and responds with text or performs audio analysis based on speech instructions. It features two interaction modes: voice chat, allowing free voice interactions without text input, and audio analysis, where users provide audio and text for analysis. The model uses natural language prompts for pre-training and has been optimized with DPO to enhance factuality and behavior adherence. Evaluation on AIR-Bench shows Qwen2-Audio outperforms previous models in audio-centric instruction-following tasks.

<hfoptions id="usage">
<hfoption id="Qwen2AudioForConditionalGeneration">

```py
import torch
from io import BytesIO
import librosa
from urllib.request import urlopen
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True, dtype="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B", trust_remote_code=True)

prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
audio, sr = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)
inputs = processor(text=prompt, audios=audio, return_tensors="pt").to(model.device)

generate_ids = model.generate(**inputs, max_length=256)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
```

</hfoption>
</hfoptions>

## Usage tips

- Voice chat mode enables free voice interactions with Qwen2-Audio without text input.
- Audio analysis mode accepts both audio and text instructions for analysis.

## Qwen2AudioConfig

[[autodoc]] Qwen2AudioConfig

## Qwen2AudioEncoderConfig

[[autodoc]] Qwen2AudioEncoderConfig

## Qwen2AudioProcessor

[[autodoc]] Qwen2AudioProcessor

## Qwen2AudioEncoder

[[autodoc]] Qwen2AudioEncoder
    - forward

## Qwen2AudioForConditionalGeneration

[[autodoc]] Qwen2AudioForConditionalGeneration
    - forward

