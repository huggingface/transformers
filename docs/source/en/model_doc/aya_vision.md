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

# AyaVision

## Overview

The Aya Vision 8B and 32B models is a state-of-the-art multilingual multimodal models developed by Cohere For AI. They build on the Aya Expanse recipe to handle both visual and textual information without compromising on the strong multilingual textual performance of the original model.

Aya Vision 8B combines the `Siglip2-so400-384-14` vision encoder with the Cohere CommandR-7B language model further post-trained with the Aya Expanse recipe, creating a powerful vision-language model capable of understanding images and generating text across 23 languages. Whereas, Aya Vision 32B uses Aya Expanse 32B as the language model.

Key features of Aya Vision include:
- Multimodal capabilities in 23 languages
- Strong text-only multilingual capabilities inherited from CommandR-7B post-trained with the Aya Expanse recipe and Aya Expanse 32B
- High-quality visual understanding using the Siglip2-so400-384-14 vision encoder
- Seamless integration of visual and textual information in 23 languages.

<!-- <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/aya_vision_architecture.webp"
alt="drawing" width="600"/>

<small> Aya Vision architecture. </small> -->

Tips:

- Aya Vision is a multimodal model that takes images and text as input and produces text as output.
- Images are represented using the `<image>` tag in the templated input.
- For best results, use the `apply_chat_template` method of the processor to format your inputs correctly.
- The model can process multiple images in a single conversation.
- Aya Vision can understand and generate text in 23 languages, making it suitable for multilingual multimodal applications.

This model was contributed by [saurabhdash](https://huggingface.co/saurabhdash) and [yonigozlan](https://huggingface.co/yonigozlan).


## Usage

Here's how to use Aya Vision for inference:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "CohereForAI/aya-vision-8b"
torch_device = "cuda:0"

# Use fast image processor
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map=torch_device, torch_dtype=torch.float16
)

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&name=medium"},
        {"type": "text", "text": "चित्र में लिखा पाठ क्या कहता है?"},
    ]},
    ]

# Process image on CUDA
inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", device=torch_device
).to(model.device)

gen_tokens = model.generate(
    **inputs, 
    max_new_tokens=300, 
    do_sample=True, 
    temperature=0.3,
)

gen_text = print(processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```
### Pipeline

```python
from transformers import pipeline

pipe = pipeline(model="CohereForAI/aya-vision-8b", task="image-text-to-text", device_map="auto")

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo="},
        {"type": "text", "text": "Bu resimde hangi anıt gösterilmektedir?"},
    ]},
    ]
outputs = pipe(text=messages, max_new_tokens=300, return_full_text=False)

print(outputs)
```

### Multiple Images and Batched Inputs

Aya Vision can process multiple images in a single conversation. Here's how to use it with multiple images:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "CohereForAI/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="cuda:0", torch_dtype=torch.float16
)

# Example with multiple images in a single message
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
            },
            {
                "type": "image",
                "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
            },
            {
                "type": "text",
                "text": "These images depict two different landmarks. Can you identify them?",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

gen_tokens = model.generate(
    **inputs, 
    max_new_tokens=300, 
    do_sample=True, 
    temperature=0.3,
)

gen_text = processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(gen_text)
```

For processing batched inputs (multiple conversations at once):

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "CohereForAI/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="cuda:0", torch_dtype=torch.float16
)

# Prepare two different conversations
batch_messages = [
    # First conversation with a single image
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                {"type": "text", "text": "Write a haiku for this image"},
            ],
        },
    ],
    # Second conversation with multiple images
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                },
                {
                    "type": "image",
                    "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
                },
                {
                    "type": "text",
                    "text": "These images depict two different landmarks. Can you identify them?",
                },
            ],
        },
    ],
]

# Process each conversation separately and combine into a batch
batch_inputs = processor.apply_chat_template(
    batch_messages, 
    padding=True, 
    add_generation_prompt=True, 
    tokenize=True, 
    return_dict=True, 
    return_tensors="pt"
).to(model.device)

# Generate responses for the batch
batch_outputs = model.generate(
    **batch_inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
)

# Decode the generated responses
for i, output in enumerate(batch_outputs):
    response = processor.tokenizer.decode(
        output[batch_inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    print(f"Response {i+1}:\n{response}\n")
```

## AyaVisionProcessor

[[autodoc]] AyaVisionProcessor

## AyaVisionConfig

[[autodoc]] AyaVisionConfig

## AyaVisionModel

[[autodoc]] AyaVisionModel

## AyaVisionForConditionalGeneration

[[autodoc]] AyaVisionForConditionalGeneration
    - forward
