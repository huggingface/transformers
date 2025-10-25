<!--Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Higgs Audio V2

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

## Overview

Higgs Audio V2 is a powerful audio foundation model developed by [Boson AI](https://www.boson.ai/). 
The model was pretrained on over 10 million hours of audio data and a diverse set of text data. 
Despite having no post-training or fine-tuning, Higgs Audio v2 excels in expressive audio generation, thanks to its deep language and acoustic understanding.

**Model Architecture:**
<div class="flex justify-center">
    <img src="https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/resolve/main/higgs_audio_v2_architecture_combined.png"/>
</div>
Higgs Audio v2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:

- Developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as AudioVerse. The in-house understanding model is finetuned on top of Higgs Audio v1 Understanding, which adopts the "understanding variant" shown in the architecture figure.
- Trained a unified audio tokenizer from scratch that captures both semantic and acoustic features.
- Proposed DualFFN architecture, which enhances the LLM’s ability to model acoustics tokens with minimal computational overhead.

## Usage Tips

### Generation with Text

```python
from transformers import HiggsAudioForConditionalGeneration, AutoProcessor, AutoTokenizer

torch_device = "cuda"

processor = AutoProcessor.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")
model = HiggsAudioForConditionalGeneration.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")

conversation = [
    {"role": "system", "content": "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"},
    {"role": "user", "content": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."},
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True).to(torch_device)

outputs = model.generate(**inputs, max_length=1024, temperature=0.3, top_p=0.95, top_k=50, tokenizer=processor.tokenizer, do_sample=True)

response = processor.decode(outputs[0], outputs[1], prompt_token_length=processor.get_prompt_len(inputs.input_ids))

processor.save_audio(response.audio, "output.wav")

```

### Generation with Text and Audio (Voice Cloning)

```python
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import HiggsAudioForConditionalGeneration, AutoProcessor, AutoTokenizer

torch_device = "cuda"

processor = AutoProcessor.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")
model = HiggsAudioForConditionalGeneration.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")

conversation = [
    {"role": "system", "content": "Generate audio following instruction with the same voice.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"},
    {"role": "user", "content": "I hear that you can understand what people say and even know their age and gender, so can you guess my age and gender from my voice?"},
    {"role": "assistant", "content": [{"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"}]},
    {"role": "user", "content": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."},
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True).to(torch_device)

outputs = model.generate(**inputs, max_length=1024, temperature=0.3, top_p=0.95, top_k=50, tokenizer=processor.tokenizer, do_sample=True)

response = processor.decode(outputs[0], outputs[1], prompt_token_length=processor.get_prompt_len(inputs.input_ids))

processor.save_audio(response.audio, "output.wav")
```

### Generation with Text and Audio (Multi-speaker Voice Cloning)

```python
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import HiggsAudioForConditionalGeneration, AutoProcessor, AutoTokenizer

torch_device = "cuda"

processor = AutoProcessor.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")
model = HiggsAudioForConditionalGeneration.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")

system_text = f"""As an AI assistant, your task is to convert written text into spoken words.
    If the user's input begins with a [SPEAKER*] tag, omit the tag and create speech based on the following content using the designated voice.
    Without a speaker tag, choose the most suitable voice for the text.
"""

conversation = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": "[SPEAKER0] I hear that you can understand what people say and even know their age and gender, so can you guess my age and gender from my voice?"},
    {"role": "assistant", "content": [{"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"}]},
    {"role": "user", "content": "[SPEAKER1] Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."},
    {"role": "assistant", "content": [{"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"}]},
    {"role": "user", "content": "[SPEAKER0] It is a doctrine of comfort and complacency, and, like all gospels, it tells its followers precisely what they wish to hear."},
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True).to(torch_device)

outputs = model.generate(**inputs, max_length=1024, temperature=0.3, top_p=0.95, top_k=50, tokenizer=processor.tokenizer, do_sample=True)

response = processor.decode(outputs[0], outputs[1], prompt_token_length=processor.get_prompt_len(inputs.input_ids))

processor.save_audio(response.audio, "output.wav")
```

### Training

```python
from io import BytesIO
from urllib.request import urlopen
import librosa
import torch
from transformers import HiggsAudioForConditionalGeneration, AutoProcessor, AutoTokenizer

torch_device = "cuda"

processor = AutoProcessor.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")
model = HiggsAudioForConditionalGeneration.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base", device_map=torch_device, torch_dtype="auto")

conversation = [
    {"role": "system", "content": "Generate audio following instruction with the same voice.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"},
    {"role": "user", "content": "I hear that you can understand what people say and even know their age and gender, so can you guess my age and gender from my voice?"},
    {"role": "assistant", "content": [{"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"}]},
    {"role": "user", "content": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."},
]
inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True).to(torch_device)

out = model(**inputs)
out.loss.backward()
```


This model was contributed by [Shuai Zheng](https://huggingface.co/szhengac). The original code can be found [here](https://github.com/boson-ai/higgs-audio).


## HiggsAudioConfig

[[autodoc]] HiggsAudioConfig

## HiggsAudioTokenizerConfig

[[autodoc]] HiggsAudioTokenizerConfig

## HiggsAudioTokenizer

[[autodoc]] HiggsAudioTokenizer
    - encode
    - decode

## HiggsAudioProcessor

[[autodoc]] HiggsAudioProcessor
    - __call__
    - decode

## HiggsAudioModel

[[autodoc]] HiggsAudioModel
    - forward

## HiggsAudioForConditionalGeneration

[[autodoc]] HiggsAudioForConditionalGeneration
    - forward
    - generate
