<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-01-10.*

# GlmImage

## Overview

GLM-Image is an image generation model adopts a hybrid autoregressive + diffusion decoder architecture, effectively pushing the upper bound of visual fidelity and fine-grained details. In general image generation quality, it aligns with industry-standard LDM-based approaches, while demonstrating significant advantages in knowledge-intensive image generation scenarios.

Model architecture: a hybrid autoregressive + diffusion decoder design、

+ Autoregressive generator: a 9B-parameter model initialized from [GLM-4-9B-0414](https://huggingface.co/zai-org/GLM-4-9B-0414), with an expanded vocabulary to incorporate visual tokens. The model first generates a compact encoding of approximately 256 tokens, then expands to 1K–4K tokens, corresponding to 1K–2K high-resolution image outputs.
+ Diffusion Decoder: a 7B-parameter decoder based on a single-stream DiT architecture for latent-space image decoding. It is equipped with a Glyph Encoder text module, significantly improving accurate text rendering within images.

Post-training with decoupled reinforcement learning: the model introduces a fine-grained, modular feedback strategy using the GRPO algorithm, substantially enhancing both semantic understanding and visual detail quality.

+ Autoregressive module: provides low-frequency feedback signals focused on aesthetics and semantic alignment, improving instruction following and artistic expressiveness.
+ Decoder module: delivers high-frequency feedback targeting detail fidelity and text accuracy, resulting in highly realistic textures, lighting, and color reproduction, as well as more precise text rendering.

GLM-Image supports both text-to-image and image-to-image generation within a single model

+ Text-to-image: generates high-detail images from textual descriptions, with particularly strong performance in information-dense scenarios.
+ Image-to-image: supports a wide range of tasks, including image editing, style transfer, multi-subject consistency, and identity-preserving generation for people and objects.

+ `GlmImageForConditionalGeneration` is the AR part of GLM-Image model, and for full image generation pipeline, please refer to [here](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/glm_image).

This model was contributed by [Raushan Turganbay](https://huggingface.co/RaushanTurganbay) and [Yuxuan Zhang](https://huggingface.co/ZHANGYUXUAN-zR).

## Usage examples

Using GLM-Image with image input to generate vision token for DIT using.

### Text-to-Image Generation

```python
from transformers import GlmImageForConditionalGeneration, AutoProcessor
import torch
import re
from math import sqrt

# Load model and processor
model_id = "zai-org/GLM-Image"
model = GlmImageForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)


def parse_shape_info(prompt: str) -> tuple[str, int, int, int, int]:
    """Parse image dimensions and expand shape tokens for two-stage generation."""
    match = re.search(r'<sop>(\d+)\s+(\d+)<eop>', prompt)
    token_h, token_w = int(match.group(1)), int(match.group(2))
    ratio = token_h / token_w
    prev_token_h = int(sqrt(ratio) * 16)
    prev_token_w = int(sqrt(1 / ratio) * 16)

    old_shape = f'<sop>{token_h} {token_w}<eop>'
    new_shape = f'<sop>{token_h} {token_w}<eop><sop>{prev_token_h} {prev_token_w}<eop>'
    expanded_prompt = prompt.replace(old_shape, new_shape)

    return expanded_prompt, token_h, token_w, prev_token_h, prev_token_w


# Text-to-Image Generation
prompt = "A cute cartoon-style text design featuring the word 'Taro' in clean, bright white rounded letters with a soft, hand-drawn feel. The background is a gentle taro purple with a misty gradient effect, decorated with small stars, hearts, and bubble elements. The overall atmosphere is light and sweet, with soft lighting like afternoon sunshine casting a warm glow from the upper left.<sop>36 24<eop>"

prompt, token_h, token_w, prev_h, prev_w = parse_shape_info(prompt)
print(f"Large image: {token_h} x {token_w} = {token_h * token_w} tokens")
print(f"Small image: {prev_h} x {prev_w} = {prev_h * prev_w} tokens")

messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# Build image grid for two-stage generation (small image + large image)
inputs["image_grid_thw"] = torch.tensor([
    [1, token_h, token_w],
    [1, prev_h, prev_w],
])

# Calculate generation parameters
small_image_tokens = prev_h * prev_w
large_image_tokens = token_h * token_w
max_new_tokens = small_image_tokens + large_image_tokens + 1

inputs = inputs.to(model.device)

# Generate image tokens
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True
)

# Extract large image tokens (skip small image tokens)
input_length = inputs["input_ids"].shape[-1]
generated_tokens = outputs[0][input_length:]
large_image_tokens_ids = generated_tokens[small_image_tokens:small_image_tokens + large_image_tokens].tolist()

print(f"Total generated tokens: {len(outputs[0]) - input_length}")
print(f"Large image tokens: {len(large_image_tokens_ids)}")
```

### Image-to-Image Generation

A portion of the Text-to-Image script can be modified—specifically the prompt and input sections—to implement
Image-to-Image generation:

```python
# Image-to-Image Generation
from PIL import Image

prompt = "Transform this image into a watercolor painting style with soft, flowing brushstrokes and pastel colors.<sop>36 24<eop>"

prompt, token_h, token_w, prev_h, prev_w = parse_shape_info(prompt)
print(f"Large image: {token_h} x {token_w} = {token_h * token_w} tokens")
print(f"Small image: {prev_h} x {prev_w} = {prev_h * prev_w} tokens")

# Load input image
image_path = "input.png"  # Replace with your image path

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_path},
            {"type": "text", "text": prompt},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# Get existing image grid from input image and append target image dimensions
existing_grid = inputs.get("image_grid_thw")
inputs["image_grid_thw"] = torch.cat([
    existing_grid,
    torch.tensor([[1, token_h, token_w]])
], dim=0)

# For image-to-image, only generate large image tokens (no small preview needed)
large_image_tokens = token_h * token_w
max_new_tokens = large_image_tokens + 1

inputs = inputs.to(model.device)

# Generate image tokens
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=True
)

# Extract generated image tokens
input_length = inputs["input_ids"].shape[-1]
generated_tokens = outputs[0][input_length:]
large_image_tokens_ids = generated_tokens[:large_image_tokens].tolist()

print(f"Total generated tokens: {len(outputs[0]) - input_length}")
print(f"Large image tokens: {len(large_image_tokens_ids)}")
```

## GlmImageConfig

[[autodoc]] GlmImageConfig

## GlmImageVisionConfig

[[autodoc]] GlmImageVisionConfig

## GlmImageTextConfig

[[autodoc]] GlmImageTextConfig

## GlmImageVQVAEConfig

[[autodoc]] GlmImageVQVAEConfig

## GlmImageImageProcessor

[[autodoc]] GlmImageImageProcessor
    - preprocess

## GlmImageImageProcessorFast

[[autodoc]] GlmImageImageProcessorFast
    - preprocess

## GlmImageProcessor

[[autodoc]] GlmImageProcessor

## GlmImageVisionModel

[[autodoc]] GlmImageVisionModel
    - forward

## GlmImageTextModel

[[autodoc]] GlmImageTextModel
    - forward

## GlmImageVQVAE

[[autodoc]] GlmImageVQVAE
    - forward

## GlmImageModel

[[autodoc]] GlmImageModel
    - forward

## GlmImageForConditionalGeneration

[[autodoc]] GlmImageForConditionalGeneration
    - forward
