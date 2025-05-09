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

# Mistral3

## Overview

Building upon Mistral Small 3 (2501), Mistral Small 3.1 (2503) adds state-of-the-art vision understanding and enhances long context capabilities up to 128k tokens without compromising text performance. With 24 billion parameters, this model achieves top-tier capabilities in both text and vision tasks.

It is ideal for:
- Fast-response conversational agents.
- Low-latency function calling.
- Subject matter experts via fine-tuning.
- Local inference for hobbyists and organizations handling sensitive data.
- Programming and math reasoning.
- Long document understanding.
- Visual understanding.

This model was contributed by [cyrilvallez](https://huggingface.co/cyrilvallez) and [yonigozlan](https://huggingface.co/yonigozlan).

The original code can be found [here](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/pixtral.py) and [here](https://github.com/mistralai/mistral-common).

## Usage example

### Inference with Pipeline

Here is how you can use the `image-text-to-text` pipeline to perform inference with the `Mistral3` models in just a few lines of code:
```python
>>> from transformers import pipeline

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
...             },
...             {"type": "text", "text": "Describe this image."},
...         ],
...     },
... ]

>>> pipe = pipeline("image-text-to-text", model="mistralai/Mistral-Small-3.1-24B-Instruct-2503", torch_dtype=torch.bfloat16)
>>> outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
>>> outputs[0]["generated_text"]
'The image depicts a vibrant and lush garden scene featuring a variety of wildflowers and plants. The central focus is on a large, pinkish-purple flower, likely a Greater Celandine (Chelidonium majus), with a'
```
### Inference on a single image

This example demonstrates how to perform inference on a single image with the Mistral3 models using chat templates.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
...             {"type": "text", "text": "Describe this image"},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> generate_ids = model.generate(**inputs, max_new_tokens=20)
>>> decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

>>> decoded_output
"The image depicts two cats lying on a pink blanket. The larger cat, which appears to be an"...
```

### Text-only generation
This example shows how to generate text using the Mistral3 model without providing any image input.


````python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = ".mistralai/Mistral-Small-3.1-24B-Instruct-2503"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

>>> SYSTEM_PROMPT = "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat."
>>> user_prompt = "Give me 5 non-formal ways to say 'See you later' in French."

>>> messages = [
...    {"role": "system", "content": SYSTEM_PROMPT},
...    {"role": "user", "content": user_prompt},
... ]

>>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
>>> inputs = processor(text=text, return_tensors="pt").to(0, dtype=torch.float16)
>>> generate_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
>>> decoded_output = processor.batch_decode(generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

>>> print(decoded_output)
"1. À plus tard!
2. Salut, à plus!
3. À toute!
4. À la prochaine!
5. Je me casse, à plus!

```
 /\_/\
( o.o )
 > ^ <
```"
````

### Batched image and text inputs
Mistral3 models also support batched image and text inputs.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
...                 {"type": "text", "text": "Describe this image"},
...             ],
...         },
...     ],
... ]


>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["Write a haiku for this imageCalm waters reflect\nWhispers of the forest's breath\nPeace on wooden path"
, "Describe this imageThe image depicts a vibrant street scene in what appears to be a Chinatown district. The focal point is a traditional Chinese"]
```

### Batched multi-image input and quantization with BitsAndBytes
This implementation of the Mistral3 models supports batched text-images inputs with different number of images for each text.
This example also how to use `BitsAndBytes` to load the model in 4bit quantization.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> quantization_config = BitsAndBytesConfig(load_in_4bit=True)
>>> model = AutoModelForImageTextToText.from_pretrained(
...     model_checkpoint, quantization_config=quantization_config
... )

>>> messages = [
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"},
...                 {"type": "image", "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"},
...                 {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
...             ],
...         },
...     ],
>>> ]

>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["Write a haiku for this imageSure, here is a haiku inspired by the image:\n\nCalm lake's wooden path\nSilent forest stands guard\n", "These images depict two different landmarks. Can you identify them? Certainly! The images depict two iconic landmarks:\n\n1. The first image shows the Statue of Liberty in New York City."]
```


## Mistral3Config

[[autodoc]] Mistral3Config

## Mistral3Model

[[autodoc]] Mistral3Model

## Mistral3ForConditionalGeneration

[[autodoc]] Mistral3ForConditionalGeneration
    - forward
