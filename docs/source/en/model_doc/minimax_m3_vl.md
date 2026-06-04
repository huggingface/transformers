<!--Copyright 2026 the HuggingFace Team. All rights reserved.

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
*This model was contributed to Hugging Face Transformers on 2026-06-03.*


# MiniMax-M3-VL

## Overview

MiniMax-M3-VL is the vision-language member of the MiniMax-M3 family. It pairs a CLIP-style vision tower (Conv3d patch embedding with 3D rotary position embeddings) with the MiniMax-M3 text backbone, a mixed dense/sparse Mixture-of-Experts decoder that uses SwiGLU-OAI gated experts and a lightning indexer for compressed sparse attention.

## Usage examples

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch


model = AutoModelForImageTextToText.from_pretrained(
    "MiniMaxAI/MiniMax-M3-preview",
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("MiniMaxAI/MiniMax-M3-preview")

image = Image.new("RGB", (672, 672), (127, 127, 127))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## MiniMaxM3VLConfig

[[autodoc]] MiniMaxM3VLConfig

## MiniMaxM3VLTextConfig

[[autodoc]] MiniMaxM3VLTextConfig

## MiniMaxM3VLVisionConfig

[[autodoc]] MiniMaxM3VLVisionConfig

## MiniMaxM3VLProcessor

[[autodoc]] MiniMaxM3VLProcessor

## MiniMaxM3VLImageProcessorFast

[[autodoc]] MiniMaxM3VLImageProcessorFast

## MiniMaxM3VLVideoProcessor

[[autodoc]] MiniMaxM3VLVideoProcessor

## MiniMaxM3VLVisionModel

[[autodoc]] MiniMaxM3VLVisionModel
    - forward

## MiniMaxM3VLTextModel

[[autodoc]] MiniMaxM3VLTextModel
    - forward

## MiniMaxM3VLModel

[[autodoc]] MiniMaxM3VLModel
    - forward

## MiniMaxM3VLForCausalLM

[[autodoc]] MiniMaxM3VLForCausalLM
    - forward

## MiniMaxM3VLForConditionalGeneration

[[autodoc]] MiniMaxM3VLForConditionalGeneration
    - forward
