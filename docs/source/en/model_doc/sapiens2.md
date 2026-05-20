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
*This model was released on 2026-04-23 and added to Hugging Face Transformers on 2026-05-19.*


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

- Sapiens2 uses Rotary Position Embeddings (RoPE) and supports arbitrary input resolutions. The default image processor resizes images to 1024×768 (height×width).
- The model uses Grouped Query Attention (GQA) for middle layers and full multi-head attention for the first and last 8 layers.
- The original checkpoint files are named `sapiens2_Xb_pretrain.safetensors` (e.g., `sapiens2_0.4b_pretrain.safetensors`) rather than `model.safetensors`. Set `transformers_weights` on the config before calling `from_pretrained` to load them correctly (see usage example below).
- Register tokens (8 by default) reduce high-norm artifacts in patch tokens, yielding cleaner attention maps and better performance on dense prediction tasks.

This model was contributed by [guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/facebookresearch/sapiens2).

## Usage examples

### Image feature extraction

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

### Semantic segmentation

The example below shows how to perform body-part segmentation with [`Sapiens2ForSemanticSegmentation`].
The segmentation checkpoints use 29 body-part labels and are named `sapiens2_0.4b_seg.safetensors`
rather than `model.safetensors`, so `transformers_weights` must be set on the config.

```python
import torch
from transformers import Sapiens2Config, Sapiens2ImageProcessor, Sapiens2ForSemanticSegmentation
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000004016.jpg"
image = load_image(url)

image_processor = Sapiens2ImageProcessor()

config = Sapiens2Config(num_labels=29)
config.transformers_weights = "sapiens2_0.4b_seg.safetensors"
model = Sapiens2ForSemanticSegmentation.from_pretrained("facebook/sapiens2-seg-0.4b", config=config)

inputs = image_processor(image, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.logits shape: (batch_size, num_labels, height, width)
print("Logits shape:", outputs.logits.shape)  # [1, 29, 1024, 768]

# Get per-pixel class predictions, optionally resized to the original image size
original_size = (image.height, image.width)
segmentation = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[original_size]
)[0]
print("Segmentation map shape:", segmentation.shape)  # [original_height, original_width]
```

### Normal estimation

The example below shows how to estimate surface normals with [`Sapiens2ForNormalEstimation`].
The normal estimation checkpoints are named `sapiens2_0.4b_normal.safetensors` rather than
`model.safetensors`, so `transformers_weights` must be set on the config. The output normals
are raw (unnormalized); use `post_process_normal_estimation` to resize and L2-normalize them.

```python
import torch
from transformers import Sapiens2Config, Sapiens2ImageProcessor, Sapiens2ForNormalEstimation
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000004016.jpg"
image = load_image(url)

image_processor = Sapiens2ImageProcessor()

config = Sapiens2Config(num_labels=3)
config.transformers_weights = "sapiens2_0.4b_normal.safetensors"
model = Sapiens2ForNormalEstimation.from_pretrained("facebook/sapiens2-normal-0.4b", config=config)

inputs = image_processor(image, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.normals shape: (batch_size, 3, height, width) — raw, unnormalized XYZ normals
print("Normals shape:", outputs.normals.shape)  # [1, 3, 1024, 768]

# Remove preprocessing padding, resize to original size, and L2-normalize to unit vectors in [-1, 1]
original_size = (image.height, image.width)
normal_maps = image_processor.post_process_normal_estimation(
    outputs, source_sizes=[original_size], target_sizes=[original_size]
)[0]
print("Normal map shape:", normal_maps.shape)   # [3, original_height, original_width]
```

## Sapiens2Config

[[autodoc]] Sapiens2Config

## Sapiens2Model

[[autodoc]] Sapiens2Model
    - forward

## Sapiens2Backbone

[[autodoc]] Sapiens2Backbone
    - forward

## Sapiens2ForSemanticSegmentation

[[autodoc]] Sapiens2ForSemanticSegmentation
    - forward

## Sapiens2ForNormalEstimation

[[autodoc]] Sapiens2ForNormalEstimation
    - forward

## Sapiens2ImageProcessor

[[autodoc]] Sapiens2ImageProcessor
    - preprocess