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

# Qwen2Audio

## Overview

The Qwen2Audio is the new model series of large audio-language models from the Qwen team.


## Usage tips

`Qwen2-Audio-7B` and `Qwen2-Audio-7B-Instruct` can be found on the [Huggingface Hub](https://huggingface.co/Qwen)

In the following, we demonstrate how to use `Qwen2-Audio-7B-Instrucct` for the inference. Note that we have used the ChatML format for dialog, in this demo we show how to leverage `apply_chat_template` for this purpose.

```python
>>> import requests
>>> from transformers import AutoProcessor, AutoModelForCausalLM
>>> from transformers.pipelines.audio_utils import ffmpeg_read
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

>>> query="<|audio_bos|><|AUDIO|><|audio_eos|>\nWhat's that?"

>>> conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': query}]

>>> text = processor.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

>>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"
>>> audio = ffmpeg_read(requests.get(url).content, sampling_rate=processor.feature_extractor.sampling_rate)

>>> inputs = processor(text=text, audios=audio, return_tensors="pt", padding=True)

>>> inputs.input_ids = inputs.input_ids.to(device)

>>> generate_ids = model.generate(**inputs, max_length=256)

>>> generate_ids = generate_ids[:, inputs.input_ids.size(1):]

>>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

## Qwen2AudioConfig

[[autodoc]] Qwen2AudioConfig

## Qwen2AudioConfig

[[autodoc]] Qwen2AudioEncoderConfig

## Qwen2AudioProcessor

[[autodoc]] Qwen2AudioProcessor

## Qwen2AudioForConditionalGeneration

[[autodoc]] Qwen2AudioForConditionalGeneration
    - forward

## Qwen2AudioEncoderModel

[[autodoc]] Qwen2AudioEncoderModel
    - forward