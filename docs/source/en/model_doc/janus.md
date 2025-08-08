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

# Janus

## Overview

The Janus Model was originally proposed in [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://huggingface.co/papers/2410.13848) by DeepSeek AI team and later refined in [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://huggingface.co/papers/2501.17811). Janus is a vision-language model that can generate both image and text output, it can also take both images and text as input.

> [!NOTE]
> The model doesn't generate both images and text in an interleaved format. The user has to pass a parameter indicating whether to generate text or image.

The abstract from the original paper is the following:

*In this paper, we introduce Janus, an autoregressive framework that unifies multimodal understanding and generation. Prior research often relies on a single visual encoder for both tasks, such as Chameleon. However, due to the differing levels of information granularity required by multimodal understanding and generation, this approach can lead to suboptimal performance, particularly in multimodal understanding. To address this issue, we decouple visual encoding into separate pathways, while still leveraging a single, unified transformer architecture for processing. The decoupling not only alleviates the conflict between the visual encoder's roles in understanding and generation, but also enhances the framework's flexibility. For instance, both the multimodal understanding and generation components can independently select their most suitable encoding methods. Experiments show that Janus surpasses previous unified model and matches or exceeds the performance of task-specific models. The simplicity, high flexibility, and effectiveness of Janus make it a strong candidate for next-generation unified multimodal models.*

The abstract from the aforementioned `Janus-Pro` paper, released afterwards, is the following:

*In this work, we introduce Janus-Pro, an advanced version of the previous work Janus. Specifically, Janus-Pro incorporates (1) an optimized training strate (2) expanded training data, and (3) scaling to larger model size. With these improvements, Janus-Pro achieves significant advancements in both multimodal understanding and text-to-image instruction-following capabilities, while also enhancing the stability of text-to-image generation. We hope this work will inspire further exploration in the field. Code and models are publicly available.*

This model was contributed by [Yaswanth Gali](https://huggingface.co/yaswanthgali) and [Hugo Silva](https://huggingface.co/hugosilva664).
The original code can be found [here](https://github.com/deepseek-ai/Janus).

## Usage Example

### Single image inference

Here is the example of visual understanding with a single image.

> [!NOTE]
> Note that the model has been trained with a specific prompt format for chatting. Use `processor.apply_chat_template(my_conversation_dict)` to correctly format your prompts.

```python
import torch
from PIL import Image
import requests

from transformers import JanusForConditionalGeneration, JanusProcessor

model_id = "deepseek-community/Janus-Pro-1B"
# Prepare Input for generation.
messages = [
    {
        "role": "user",
        "content": [
            {'type':'image', 'url': 'http://images.cocodataset.org/val2017/000000039769.jpg'},
            {'type':"text", "text":"What do you see in this image?."}
        ]
    },
]

# Set generation mode to `text` to perform text generation.
processor = JanusProcessor.from_pretrained(model_id)
model = JanusForConditionalGeneration.from_pretrained(model_id,     
        dtype=torch.bfloat16,
        device_map="auto")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    generation_mode="text",
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output = model.generate(**inputs, max_new_tokens=40,generation_mode='text',do_sample=True)
text = processor.decode(output[0], skip_special_tokens=True)
print(text)
```

### Multi image inference

Janus can perform inference with multiple images as input, where images can belong to the same prompt or different prompts in batched inference, where the model processes many conversations in parallel. Here is how you can do it:

```python
import torch
from PIL import Image
import requests

from transformers import JanusForConditionalGeneration, JanusProcessor

model_id = "deepseek-community/Janus-Pro-1B"

image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://www.ilankelman.org/stopsigns/australia.jpg",
    "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
]

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s the difference between"},
                {"type": "image", "url": image_urls[0]},
                {"type": "text", "text": " and "},
                {"type": "image", "url": image_urls[1]}
            ]
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_urls[2]},
                {"type": "text", "text": "What do you see in this image?"}
            ]
        }
    ]
]

# Load model and processor
processor = JanusProcessor.from_pretrained(model_id)
model = JanusForConditionalGeneration.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map="auto"
)

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    generation_mode="text",
    tokenize=True,
    padding=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

# Generate response
output = model.generate(**inputs, max_new_tokens=40, generation_mode='text', do_sample=False)
text = processor.batch_decode(output, skip_special_tokens=True)
print(text)
```

## Text to Image generation

Janus can also generate images given a prompt.

```python
import torch
from transformers import JanusForConditionalGeneration, JanusProcessor

# Set generation mode to `image` to prepare inputs for image generation..

model_id = "deepseek-community/Janus-Pro-1B"
processor = JanusProcessor.from_pretrained(model_id)
model = JanusForConditionalGeneration.from_pretrained(model_id,
        dtype=torch.bfloat16,
        device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "A dog running under the rain."},
        ],
     }
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt,generation_mode="image",return_tensors="pt").to(model.device, dtype=torch.bfloat16)

# Set num_return_sequence parameter to generate multiple images per prompt.
model.generation_config.num_return_sequences = 2
outputs = model.generate(**inputs,
                         generation_mode="image",
                         do_sample=True,
                         use_cache=True,
                         )
# Perform post-processing on the generated token ids.
decoded_image = model.decode_image_tokens(outputs)
images = processor.postprocess(list(decoded_image.float()),return_tensors="PIL.Image.Image")
# Save the image
for i, image in enumerate(images['pixel_values']):
    image.save(f"result{i}.png")
```

## JanusConfig

[[autodoc]] JanusConfig

## JanusVisionConfig

[[autodoc]] JanusVisionConfig

## JanusVQVAEConfig

[[autodoc]] JanusVQVAEConfig

## JanusProcessor

[[autodoc]] JanusProcessor

## JanusImageProcessor

[[autodoc]] JanusImageProcessor

## JanusImageProcessorFast

[[autodoc]] JanusImageProcessorFast

## JanusVisionModel

[[autodoc]] JanusVisionModel
    - forward

## JanusVQVAE

[[autodoc]] JanusVQVAE
    - forward

## JanusModel

[[autodoc]] JanusModel
    - forward

## JanusForConditionalGeneration

[[autodoc]] JanusForConditionalGeneration
    - forward
