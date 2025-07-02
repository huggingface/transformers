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

# Dia

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

Dia is an opensource text-to-speech (TTS) model (1.6B parameters) developed by [Nari Labs](https://huggingface.co/nari-labs).
It can generate highly realistic dialogue from transcript including nonverbal communications such as laughter and coughing.
Furthermore, emotion and tone control is also possible via audio conditioning (voice cloning).

**Model Architecture:**
Dia is an encoder-decoder transformer based on the original transformer architecture. However, some more modern features such as
rotational positional embeddings (RoPE) are also included. For its text portion (encoder), a byte tokenizer is utilized while
for the audio portion (decoder), a pretrained codec model [DAC](./dac.md) is used - DAC encodes speech into discrete codebook
tokens and decodes them back into audio.

## Usage Tips

### Generation with Text

```python
from transformers import AutoProcessor, DiaForConditionalGeneration

torch_device = "cuda"
model_checkpoint = "buttercrab/dia-v1-1.6b"

text = ["[S1] Dia is an open weights text to dialogue model."]
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=256)  # corresponds to around ~2s

# save audio to a file
outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example.wav")

```

### Generation with Text and Audio (Voice Cloning)

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration

torch_device = "cuda"
model_checkpoint = "buttercrab/dia-v1-1.6b"

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio = ds[-1]["audio"]["array"]
# text is a transcript of the audio + additional text you want as new audio
text = ["[S1] I know. It's going to save me a lot of money, I hope. [S2] I sure hope so for you."]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, audio=audio, padding=True, return_tensors="pt").to(torch_device)
prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=256)  # corresponds to around ~2s

# retrieve actually generated audio and save to a file
outputs = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
processor.save_audio(outputs, "example_with_audio.wav")
```

### Training

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration

torch_device = "cuda"
model_checkpoint = "buttercrab/dia-v1-1.6b"

ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio = ds[-1]["audio"]["array"]
# text is a transcript of the audio
text = ["[S1] I know. It's going to save me a lot of money, I hope."]

processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(
    text=text,
    audio=audio,
    generation=False,
    output_labels=True,
    padding=True,
    return_tensors="pt"
).to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
out = model(**inputs)
out.loss.backward()
```


This model was contributed by [Jaeyong Sung](https://huggingface.co/buttercrab), [Arthur Zucker](https://huggingface.co/ArthurZ),
and [Anton Vlasjuk](https://huggingface.co/AntonV). The original code can be found [here](https://github.com/nari-labs/dia/).


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
