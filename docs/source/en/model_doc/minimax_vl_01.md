<!--Copyright 2026 MiniMaxAI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-04.*

# MiniMax-VL-01

## Overview

[MiniMax-VL-01](https://huggingface.co/MiniMaxAI/MiniMax-VL-01) is the vision-language model introduced in
[MiniMax-01: Scaling Foundation Models with Lightning Attention](https://huggingface.co/papers/2501.08313). It combines
a 24-layer CLIP vision encoder, a two-layer GELU projector, and the 456B-parameter MiniMax-Text-01 language model.
The language model alternates seven Lightning Attention layers with one full-attention layer and uses a top-2
Mixture-of-Experts feed-forward network in every layer.

Images are processed at dynamic resolutions. Each image is represented by a 336×336 thumbnail and a grid of
336×336 crops chosen for its aspect ratio. The projected crop features are reassembled spatially, unpadded, and
separated by learned newline embeddings before replacing the expanded `<image>` tokens in the text prompt.

The original project publishes its code under the MIT license and the model materials under the separate
[MiniMax Model License](https://github.com/MiniMax-AI/MiniMax-01/blob/main/LICENSE-MODEL). Review the model license
before downloading or using the weights.

> [!WARNING]
> MiniMax-VL-01 is exceptionally large. The unquantized checkpoint contains about 456B parameters and requires
> roughly 913 GB just to store the weights in 16-bit precision. Check available storage and accelerator memory before
> downloading it.

## Usage

Use the processor's chat template so that image placeholders and role markers match the checkpoint. The processor
expands each image placeholder to the exact number of visual features produced for that image.

```python
import torch

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


checkpoint = "MiniMaxAI/MiniMax-VL-01"
processor = AutoProcessor.from_pretrained(checkpoint)
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint,
    dtype=torch.bfloat16,
    device_map="auto",
)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## MiniMaxVL01Config

[[autodoc]] MiniMaxVL01Config

## MiniMaxVL01ImageProcessor

[[autodoc]] MiniMaxVL01ImageProcessor
    - preprocess

## MiniMaxVL01ImageProcessorPil

[[autodoc]] MiniMaxVL01ImageProcessorPil
    - preprocess

## MiniMaxVL01Processor

[[autodoc]] MiniMaxVL01Processor
    - __call__

## MiniMaxVL01Model

[[autodoc]] MiniMaxVL01Model
    - forward
    - get_image_features

## MiniMaxVL01ForConditionalGeneration

[[autodoc]] MiniMaxVL01ForConditionalGeneration
    - forward
    - get_image_features
