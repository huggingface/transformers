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
*This model was released on 2025-04-21 and added to Hugging Face Transformers on 2025-06-26 and contributed by [buttercrab](https://huggingface.co/buttercrab) and [ArthurZ](https://huggingface.co/ArthurZ).*

# Dia

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[Dia](https://github.com/nari-labs/dia) is a 1.6B-parameter text-to-speech model from Nari Labs designed to generate natural, emotionally expressive dialogue, including non-verbal sounds like laughter and coughing. It uses an encoder-decoder transformer architecture enhanced with modern features such as rotational positional embeddings (RoPE). Text input is processed with a byte tokenizer, while audio is handled through a pretrained DAC codec that converts speech to and from discrete codebook tokens. This setup enables realistic voice synthesis with controllable tone and emotion via audio conditioning.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-to-audio", model="nari-labs/Dia-1.6B-0626", dtype="auto")
output = pipeline("Plants create energy through a process known as photosynthesis.")
audio = output["audio"]
```

</hfoption>
<hfoption id="DiaForConditionalGeneration">

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration

processor = AutoProcessor.from_pretrained("nari-labs/Dia-1.6B-0626")
model = DiaForConditionalGeneration.from_pretrained("nari-labs/Dia-1.6B-0626").to(torch_device)

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio = ds[-1]["audio"]["array"]
text = ["[S1] Plants create energy through a process known as photosynthesis. [S2] That is so amazing!"]
inputs = processor(text=text, audio=audio, padding=True, return_tensors="pt")
prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

outputs = model.generate(**inputs, max_new_tokens=256)
outputs = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
processor.save_audio(outputs, "example_with_audio.wav")
```

</hfoption>
</hfoptions>

## DiaConfig

[[autodoc]] DiaConfig

## DiaDecoderConfig

[[autodoc]] DiaDecoderConfig

## DiaEncoderConfig

[[autodoc]] DiaEncoderConfig

## DiaTokenizer

[[autodoc]] DiaTokenizer
    - __call__

## DiaFeatureExtractor

[[autodoc]] DiaFeatureExtractor
    - __call__

## DiaProcessor

[[autodoc]] DiaProcessor
    - __call__
    - batch_decode
    - decode

## DiaModel

[[autodoc]] DiaModel
    - forward

## DiaForConditionalGeneration

[[autodoc]] DiaForConditionalGeneration
    - forward
    - generate
