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
*This model was published in HF papers on 2024-12-13 and contributed to Hugging Face Transformers on 2026-11-20.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DeepseekVLV2

[DeepSeek-VL2](https://huggingface.co/papers/2412.10302) is a family of Mixture-of-Experts vision-language models from DeepSeek AI. A [SigLIP](./siglip) so400m encoder reads each image twice: once as a padded global view, and once as 384 x 384 tiles cut from the best of 23 candidate canvases, so high-resolution and extreme-aspect images keep their detail. A projector merges every 2 x 2 block of patches into one token, and learned newline and view-separator tokens tell the language model where each row and each view ends.

The three released sizes share that vision side and differ in the language model:

| Checkpoint | Language model | Attention | Routing |
|---|---|---|---|
| deepseek-vl2-tiny (3.37B) | DeepSeekMoE, 12 layers | multi-head | greedy top-6 of 64 experts |
| deepseek-vl2-small (16.1B) | [DeepSeek-V2](./deepseek_v2) | MLA | greedy top-6 of 64 experts |
| deepseek-vl2 (27.5B) | [DeepSeek-V3](./deepseek_v3) style | MLA | sigmoid top-6 of 72 experts with bias correction |

The example below generates a description of an image.

```python
import torch
from transformers import AutoProcessor, DeepseekVLV2ForConditionalGeneration

model_id = "deepseek-community/deepseek-vl2-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = DeepseekVLV2ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device, dtype=model.dtype)

generated = model.generate(**inputs, max_new_tokens=64)
print(processor.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Notes

- The image processor returns three tensors: `pixel_values` (one global view per image), `pixel_values_local` (all local tiles, row by row, concatenated across images) and `image_spatial_crop` (`[width_tiles, height_tiles]` per image). Pass all three to the model.
- The original model tiles an image only when the prompt has at most two images; for more, call the processor with `crop_to_patches=False` so each image gets a single local tile, as in training.
- Each image takes `14 * 15 + 1 + (14 * height_tiles) * (14 * width_tiles + 1)` tokens, so a 2 x 2 tiling costs 211 + 28 * 29 = 1023 tokens.

## DeepseekVLV2Config

[[autodoc]] DeepseekVLV2Config

## DeepseekVLV2Processor

[[autodoc]] DeepseekVLV2Processor
    - __call__

## DeepseekVLV2ImageProcessor

[[autodoc]] DeepseekVLV2ImageProcessor
    - preprocess

## DeepseekVLV2Model

[[autodoc]] DeepseekVLV2Model
    - forward

## DeepseekVLV2ForConditionalGeneration

[[autodoc]] DeepseekVLV2ForConditionalGeneration
    - forward
