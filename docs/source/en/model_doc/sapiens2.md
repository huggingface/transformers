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
*This model was released on 2026-04-23 and added to Hugging Face Transformers on 2026-05-14.*


# Sapiens2

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

## Overview

The Sapiens2 model was proposed in [Sapiens2](https://huggingface.co/papers/2604.21681) by Rawal Khirodkar, He Wen, Julieta Martinez, Yuan Dong, Zhaoen Su, Shunsuke Saito.
Sapiens2 is a family of high-resolution vision transformers pretrained on ~1 billion curated human images, designed for human-centric computer vision tasks including pose estimation, body-part segmentation, surface normal estimation, and pointmap estimation.

You can find all the original Sapiens2 checkpoints under the [Sapiens2](https://huggingface.co/collections/facebook/sapiens2) collection.

The abstract from the paper is the following:

*We present Sapiens2, a family of high-resolution transformers for human-centric vision focused on generalization, versatility, and high-fidelity outputs. We pretrain on ~1 billion curated high-quality human images with improved task annotations and combine masked image reconstruction with self-distilled contrastive objectives to learn both low-level and semantic features. Our models scale from 0.4B to 5B parameters and train at native 1K resolution, with hierarchical 4K variants for extended spatial reasoning. Sapiens2 achieves substantial improvements over its predecessor: +4 mAP in pose estimation, +24.3 mIoU in body-part segmentation, and 45.6% error reduction in normal estimation, while extending to new tasks like pointmap and albedo estimation. Code is publicly available.*

Tips:

- Sapiens2 uses Rotary Position Embeddings (RoPE) and supports arbitrary input resolutions. The default image processor resizes images to 768×1024.
- The model uses Grouped Query Attention (GQA) for middle layers and full multi-head attention for the first and last 8 layers.
- The original checkpoint files are named `sapiens2_Xb_pretrain.safetensors` (e.g., `sapiens2_0.4b_pretrain.safetensors`) rather than `model.safetensors`. Set `transformers_weights` on the config before calling `from_pretrained` to load them correctly (see usage example below).
- Register tokens (8 by default) reduce high-norm artifacts in patch tokens, yielding cleaner attention maps and better performance on dense prediction tasks.

This model was contributed by [guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/facebookresearch/sapiens2).

## Usage examples

The example below shows how to obtain image features with [`Sapiens2Model`]. The original checkpoint
files are named `sapiens2_0.4b_pretrain.safetensors` rather than `model.safetensors`, so you must set
`transformers_weights` on the config to point `from_pretrained` to the correct file.

```python
import torch
from transformers import Sapiens2Config, Sapiens2ImageProcessor, Sapiens2Model
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000004016.jpg"
image = load_image(url)

image_processor = Sapiens2ImageProcessor()

config = Sapiens2Config()
config.transformers_weights = "sapiens2_0.4b_pretrain.safetensors"
model = Sapiens2Model.from_pretrained("facebook/sapiens2-pretrain-0.4b", config=config)

inputs = image_processor(image, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.pooler_output is the CLS token (whole-image embedding)
cls_token = outputs.pooler_output

# Split patch tokens from last_hidden_state for dense tasks
_, _, height, width = inputs["pixel_values"].shape
num_patches_h = height // model.config.patch_size
num_patches_w = width // model.config.patch_size
patch_tokens = outputs.last_hidden_state[:, 1 + model.config.num_register_tokens :, :]
patch_features = patch_tokens.unflatten(1, (num_patches_h, num_patches_w))

print("CLS token shape:", cls_token.shape)           # [1, 1024]
print("Patch features shape:", patch_features.shape) # [1, H/patch, W/patch, 1024]
```

## Sapiens2Config

[[autodoc]] Sapiens2Config

## Sapiens2Model

[[autodoc]] Sapiens2Model
    - forward

## Sapiens2Backbone

[[autodoc]] Sapiens2Backbone
    - forward

## Sapiens2ImageProcessor

[[autodoc]] Sapiens2ImageProcessor
    - preprocess