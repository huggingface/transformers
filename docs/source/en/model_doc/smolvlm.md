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

# SmolVLM

## Overview
SmolVLM2 is an adaptation of the Idefics3 model with three main differences:

- It uses SmolLM2 for the text model.
- It supports multi-image and video inputs

## Usage tips

Input images are processed either by upsampling (if resizing is enabled) or at their original resolution. The resizing behavior depends on two parameters: do_resize and size.

Videos should not be upsampled. 

If `do_resize` is set to `True`, the model resizes images so that the longest edge is 4*364 pixels by default.
The default resizing behavior can be customized by passing a dictionary to the `size` parameter. For example, `{"longest_edge": 4 * 364}` is the default, but you can change it to a different value if needed.

Here’s how to control resizing and set a custom size:
```python
image_processor = SmolVLMImageProcessor(do_resize=True, size={"longest_edge": 2 * 512}, max_image_size=512)
```

Additionally, the `max_image_size` parameter, which controls the size of each square patch the image is decomposed into, is set to 512 by default but can be adjusted as needed. After resizing (if applicable), the image processor decomposes the images into square patches based on the `max_image_size` parameter.

This model was contributed by [orrzohar](https://huggingface.co/orrzohar).



## Usage example

### Single Media inference

The model can accept both images and videos as input, but you should use only one of the modalities within one inference. Here's an example code for that.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

conversation = [
    {
        "role":"user",
        "content":[
            {"type":"image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type":"text", "text":"Describe this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=128)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)


# Video
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/path/to/video.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

### Batch Mixed Media Inference

The model can batch inputs composed of several images/videos and text. Here is an example.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Conversation for the first image
conversation1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "Describe this image."}
        ]
    }
]

# Conversation with two images
conversation2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "image", "path": "/path/to/image.jpg"},
            {"type": "text", "text": "What is written in the pictures?"}
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {"role": "user","content": "who are you?"}
]


conversations = [conversation1, conversation2, conversation3]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

## SmolVLMConfig

[[autodoc]] SmolVLMConfig

## SmolVLMVisionConfig

[[autodoc]] SmolVLMVisionConfig

## Idefics3VisionTransformer

[[autodoc]] SmolVLMVisionTransformer

## SmolVLMModel

[[autodoc]] SmolVLMModel
    - forward

## SmolVLMForConditionalGeneration

[[autodoc]] SmolVLMForConditionalGeneration
    - forward


## SmolVLMImageProcessor
[[autodoc]] SmolVLMImageProcessor
    - preprocess


## SmolVLMProcessor
[[autodoc]] SmolVLMProcessor
    - __call__
