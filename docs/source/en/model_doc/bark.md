<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2023-07-17 and contributed by [ylacombe](https://huggingface.co/ylacombe) and [sanchit-gandhi](https://github.com/sanchit-gandhi).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# Bark

[Bark](https://github.com/suno-ai/bark) is a text-to-audio generative model capable of producing realistic speech, music, and sound effects directly from text prompts. It’s built using a transformer-based architecture that models audio tokens rather than phonemes, enabling it to capture tone, emotion, and multilingual speech without explicit linguistic preprocessing. Bark uses semantic and coarse acoustic tokens, trained on diverse multilingual datasets, to generate natural prosody and expressive delivery. Its outputs are decoded from discrete audio representations, similar in spirit to models like EnCodec or VALL-E, allowing highly expressive and context-aware audio synthesis.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-to-audio", model="suno/bark-small", dtype="auto")
output = pipeline("Plants create energy through a process known as photosynthesis.")
audio = output["audio"]
```

</hfoption>
<hfoption id="BarkModel">

```py
import torch
from scipy.io.wavfile import write as write_wav
from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark", dtype="auto")

inputs = processor("Plants create energy through a process known as photosynthesis.", voice_preset="v2/en_speaker_6")
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()
sample_rate = model.generation_config.sample_rate
write_wav("bark_generation.wav", sample_rate, audio_array)
```

</hfoption>
</hfoptions>

## Usage tips

- Suno provides voice presets in multiple languages. Find them in the [official library](https://github.com/suno-ai/bark/tree/main/notebooks) or on the [Hugging Face Hub](https://huggingface.co/suno/bark/tree/main/notebooks).
- Bark generates highly realistic, multilingual speech plus other audio types. This includes music, background noise, and simple sound effects. The model also produces nonverbal communications like laughing, sighing, and crying.

## BarkConfig

[[autodoc]] BarkConfig
    - all

## BarkProcessor

[[autodoc]] BarkProcessor
    - all
    - __call__

## BarkModel

[[autodoc]] BarkModel
    - generate
    - enable_cpu_offload

## BarkSemanticModel

[[autodoc]] BarkSemanticModel
    - forward

## BarkCoarseModel

[[autodoc]] BarkCoarseModel
    - forward

## BarkFineModel

[[autodoc]] BarkFineModel
    - forward

## BarkCausalModel

[[autodoc]] BarkCausalModel
    - forward

## BarkCoarseConfig

[[autodoc]] BarkCoarseConfig
    - all

## BarkFineConfig

[[autodoc]] BarkFineConfig
    - all

## BarkSemanticConfig

[[autodoc]] BarkSemanticConfig
    - all

