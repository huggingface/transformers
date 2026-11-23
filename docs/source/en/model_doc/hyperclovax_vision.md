<!--Copyright 2026 NAVER Corp. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-23.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# HyperCLOVAX Vision

HyperCLOVAX Vision is a lightweight vision-language model developed by NAVER, with strong support for Korean. It follows the [LLaVA-NeXT](./llava_next) design: a [SigLIP](./siglip) vision encoder and a [Llama](./llama) language model are connected by a C-Abstractor projector from [Honeybee](https://github.com/kakaobrain/honeybee), which reduces every 378x378 image grid to 81 visual tokens with convolutional blocks and adaptive pooling. High resolution images are split into up to 9 grids (AnyRes), next to a global view of the whole image that keeps its aspect ratio.

You can find the original checkpoint on the [naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B) page.

The example below demonstrates how to generate text based on an image with [`HyperCLOVAXVisionForConditionalGeneration`].

```python
import torch
from transformers import AutoProcessor, HyperCLOVAXVisionForConditionalGeneration

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"
model = HyperCLOVAXVisionForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device, torch.bfloat16)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Notes

- The number of image tokens depends on the size of each image. The processor expands every image token in the prompt to the right number of tokens, so the prompt should contain a single image token per image.
- Video inputs aren't supported yet.

## HyperCLOVAXVisionConfig

[[autodoc]] HyperCLOVAXVisionConfig

## HyperCLOVAXVisionImageProcessor

[[autodoc]] HyperCLOVAXVisionImageProcessor
    - preprocess

## HyperCLOVAXVisionImageProcessorPil

[[autodoc]] HyperCLOVAXVisionImageProcessorPil
    - preprocess

## HyperCLOVAXVisionProcessor

[[autodoc]] HyperCLOVAXVisionProcessor
    - __call__

## HyperCLOVAXVisionModel

[[autodoc]] HyperCLOVAXVisionModel
    - forward
    - get_image_features

## HyperCLOVAXVisionForConditionalGeneration

[[autodoc]] HyperCLOVAXVisionForConditionalGeneration
    - forward
    - get_image_features
