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

# InternVL

## Overview

The InternVL2.5-MPO family of models was introduced in [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://huggingface.co/papers/2411.10442) by Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao and Jifeng Dai.

The InternVL2.5-MPO family of models is a state-of-the-art series of multimodal large language models built to improve reasoning and understanding across text, images and videos.
A key innovation behind these models is Mixed Preference Optimization (MPO), a training approach that strengthens multimodal reasoning by leveraging a specialized Multi-Modal Preference Dataset (MMPR) with 3 million carefully curated examples. By balancing preference loss, quality loss, and generation loss, MPO helps the models generate more relevant, coherent, and insightful responses.
To handle visual inputs more effectively, the InternVL2.5-MPO models also incorporate techniques like pixel unshuffle operations and dynamic resolution strategies, ensuring efficient processing without sacrificing detail.

The abstract from the paper is the following:

*Existing open-source multimodal large language models (MLLMs) generally follow a training process involving pre-training and supervised fine-tuning. However, these models suffer from distribution shifts, which limit their multimodal reasoning, particularly in the Chain-of-Thought (CoT) performance. To address this, we introduce a preference optimization (PO) process to enhance the multimodal reasoning capabilities of MLLMs. Specifically, (1) on the data side, we design an automated preference data construction pipeline to create MMPR, a high-quality, large-scale multimodal reasoning preference dataset. and (2) on the model side, we explore integrating PO with MLLMs, developing a simple yet effective method, termed Mixed Preference Optimization (MPO), which boosts multimodal CoT performance. Our approach demonstrates improved performance across multiple benchmarks, particularly in multimodal reasoning tasks. Notably, our model, InternVL2-8B-MPO, achieves an accuracy of 67.0 on MathVista, outperforming InternVL2-8B by 8.7 points and achieving performance comparable to the 10x larger InternVL2-76B. We hope this study could inspire further advancements in MLLMs. Code, data, and model shall be publicly released.*


<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/internvl_architecture.png" alt="drawing" width="600"/>

<small> Overview of InternVL2.5-MPO models architecture. Taken from the <a href="https://huggingface.co/OpenGVLab/InternVL2_5-1B-MPO">original checkpoint.</a> </small>



<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/internvl_overview_performance.png" alt="drawing" width="600"/>

<small> Comparison of InternVL2.5-MPO performance on OpenCompass against other SOTA VLLMs. Taken from the <a href="https://huggingface.co/OpenGVLab/InternVL2_5-1B-MPO">original checkpoint.</a> </small>



This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/OpenGVLab/InternVL).

## Usage example

### Inference with Pipeline

Here is how you can use the `image-text-to-text` pipeline to perform inference with the `InternVL2.5-MPO` models in just a few lines of code:
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

>>> pipe = pipeline("image-text-to-text", model="../InternVLTest-1B")
>>> outputs = pipe(text=messages, max_new_tokens=50, return_full_text=False)
>>> outputs[0]["generated_text"]
'The image depicts a close-up view of a vibrant garden scene, featuring several flowers in various stages of bloom. The primary focus is on a cluster of pink cosmos flowers, which are characterized by their large, five-petaled petals and prominent yellow centers'
```
### Inference on a single image

This example demonstrates how to perform inference on a single image with the InternVL models using chat templates.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "../InternVLTest-1B"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
...             {"type": "text", "text": "Please describe the image explicitly."},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

>>> generate_ids = model.generate(**inputs, max_new_tokens=50)
>>> decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

>>> decoded_output
'The image shows two cats lying on a pink couch. One cat is on the left side of the couch, and the other is on the right. Both cats are sleeping. Between the two cats, there are two remote controls, one on each side'
```

### Text-only generation
This example shows how to generate text using the InternVL model without providing any image input.


```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "../InternVLTest-1B"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {"type": "text", "text": "Write a haiku"},
...         ],
...     }
... ]

>>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)

>>> generate_ids = model.generate(**inputs, max_new_tokens=50)
>>> decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

>>> print(decoded_output)
'Whispers of dawn,\nLeaves rustle in the breeze,\nNew day begins.'
```

### Batched image and text inputs
InternVL models also support batched image and text inputs.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "../InternVLTest-1B"
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


>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["user\n\nWrite a haiku for this image\nassistant\nA pier stretches to the horizon,\nIn misty mountains, nature's still,\nPeace awaits beneath the sky.",
 'user\n\nDescribe this image\nassistant\nThe image shows a street scene with a traditional Chinese gate in the background, featuring red pillars and ornate decorations. There is']
```

### Batched multi-image input
This implementation of the InternVL models supports batched text-images inputs with different number of images for each text.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "../InternVLTest-1B"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

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

>>> inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
>>> decoded_outputs
["user\n\nWrite a haiku for this image\nassistant\nA pier stretches to the horizon,\nIn the stillness of the lake, a quiet dream,\nNature's peace unfolds.",
 'user\n\n\nThese images depict two different landmarks. Can you identify them?\nassistant\nThe images depict the Statue of Liberty and the Golden Gate Bridge.']
```

### Video input
InternVL models can also handle video inputs. Here is an example of how to perform inference on a video input using chat templates.

TODO modify video loading

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

>>> model_checkpoint = "../InternVLTest-4B"
>>> quantization_config = BitsAndBytesConfig(load_in_4bit=True)
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, quantization_config=quantization_config)

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "video",
...                 "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
...             },
...             {"type": "text", "text": "What type of shot is the man performing?"},
...         ],
...     }
>>> ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     return_tensors="pt",
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     num_frames=8,
...     initial_shift=True,
>>> ).to(model.device)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_output = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
>>> decoded_output
'The man is performing a forehand stroke. This is evident from his stance and the position of his body relative to the net'
```

### Interleaved image and video inputs
This example showcases how to handle a batch of chat conversations with interleaved image and video inputs using chat template.

```python
>>> from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
>>> import torch

>>> torch_device = "cuda"
>>> model_checkpoint = "../InternVLTest-1B"
>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
>>> model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

>>> messages = [
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
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
...                 {"type": "text", "text": "What type of shot is the man performing?"},
...             ],
...         },
...     ],
...     [
...         {
...             "role": "user",
...             "content": [
...                 {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
...                 {"type": "text", "text": "Write a haiku for this image"},
...             ],
...         },
...     ],
>>> ]
>>> inputs = processor.apply_chat_template(
...     messages,
...     padding=True,
...     add_generation_prompt=True,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
...     num_frames=8,
...     initial_shift=True,
>>> ).to(model.device)

>>> output = model.generate(**inputs, max_new_tokens=25)

>>> decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
>>> decoded_outputs
['user\n\n\nThese images depict two different landmarks. Can you identify them?\nassistant\nThe images depict the Statue of Liberty and the Golden Gate Bridge.',
 'user\nFrame1: \nFrame2: \nFrame3: \nFrame4: \nFrame5: \nFrame6: \nFrame7: \nFrame8: \nWhat type of shot is the man performing?\nassistant\nThe man is performing a forehand shot.',
 "user\n\nWrite a haiku for this image\nassistant\nA pier stretches to the horizon,\nIn the stillness, nature's calm is found,\nPeace awaits beneath the sky."]
```

## InternVLVisionConfig

[[autodoc]] InternVLVisionConfig

## InternVLConfig

[[autodoc]] InternVLConfig

## InternVLVisionModel

[[autodoc]] InternVLVisionModel
    - forward

## InternVLForConditionalGeneration

[[autodoc]] InternVLForConditionalGeneration
    - forward

## InternVLProcessor

[[autodoc]] InternVLProcessor
