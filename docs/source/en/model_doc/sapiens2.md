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
*This model was published in HF papers on 2026-04-23 and contributed to Hugging Face Transformers on 2026-06-01.*


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
- Register tokens (8 by default) reduce high-norm artifacts in patch tokens, yielding cleaner attention maps and better performance on dense prediction tasks.

This model was contributed by [guarin](https://huggingface.co/guarin).
The original code can be found [here](https://github.com/facebookresearch/sapiens2).

## Usage examples

<hfoptions id="usage">

<hfoption id="AutoModel">

The example below shows how to obtain the CLS token (whole-image embedding) with [`Sapiens2Model`].

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pretrain-0.4b", revision="refs/pr/1")
model = AutoModel.from_pretrained("facebook/sapiens2-pretrain-0.4b", device_map="auto", revision="refs/pr/1")

inputs = image_processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.pooler_output is the CLS token (whole-image embedding)
cls_token = outputs.pooler_output
print("CLS token shape:", cls_token.shape)  # [1, 1024]
```

</hfoption>

<hfoption id="AutoBackbone">

[`Sapiens2Backbone`] exposes patch tokens already reshaped to spatial dimensions and CLS tokens directly on the output object.

```python
import torch
from transformers import AutoBackbone, AutoImageProcessor
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pretrain-0.4b", revision="refs/pr/1")
model = AutoBackbone.from_pretrained("facebook/sapiens2-pretrain-0.4b", device_map="auto", revision="refs/pr/1")

inputs = image_processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs, return_class_token=True)

# Patch tokens shaped (batch, height, width, channels)
patch_features = outputs.feature_maps[0]
cls_token = outputs.cls_tokens[0]
print("CLS token shape:", cls_token.shape)           # [1, 1024]
print("Patch features shape:", patch_features.shape) # [1, 64, 48, 1024]
```

</hfoption>

<hfoption id="Normal estimation">

The example below shows how to estimate surface normals with [`Sapiens2ForNormalEstimation`].
The output normals are raw (unnormalized); use `post_process_normal_estimation` to resize and L2-normalize them.

```python
import torch
from transformers import AutoImageProcessor, AutoModelForNormalEstimation
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-normal-0.4b", revision="refs/pr/1")
model = AutoModelForNormalEstimation.from_pretrained("facebook/sapiens2-normal-0.4b", device_map="auto", revision="refs/pr/1")

inputs = image_processor(image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.normals shape: (batch_size, 3, height, width) — raw, unnormalized XYZ normals
print("Normals shape:", outputs.normals.shape)  # [1, 3, 1024, 768]

# Remove preprocessing padding, resize to original size, and L2-normalize to unit vectors in [-1, 1]
original_size = (image.height, image.width)
result = image_processor.post_process_normal_estimation(
    outputs, source_sizes=[original_size], target_sizes=[original_size]
)
normals = result[0]["normals"]
print("Normals shape:", normals.shape)   # [3, original_height, original_width]

# Convert L2-normalized normals in [-1, 1] to RGB in [0, 255]
normals_rgb = ((normals + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)

# Apply background removal using the segmentation model output.
# `segmentation` is the output of `post_process_semantic_segmentation` — a (H, W) tensor
# of per-pixel class IDs, where class 0 is background.
background_mask = segmentation == 0
normals_rgb[:, background_mask] = 0
print("Normals RGB shape:", normals_rgb.shape)   # [3, original_height, original_width]
```

</hfoption>

<hfoption id="Pointmap estimation">

The example below shows how to estimate per-pixel 3D coordinates with [`Sapiens2ForPointmapEstimation`].
Use `post_process_pointmap_estimation` to remove preprocessing padding, resize to the original image size, and apply the predicted focal-length scale.

```python
import torch
from transformers import AutoImageProcessor, AutoModelForPointmapEstimation
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pointmap-0.4b", revision="refs/pr/1")
model = AutoModelForPointmapEstimation.from_pretrained("facebook/sapiens2-pointmap-0.4b", device_map="auto", revision="refs/pr/1")

inputs = image_processor(image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.pointmaps shape: (batch_size, 3, height, width) — raw XYZ in canonical camera space
print("Pointmaps shape:", outputs.pointmaps.shape)  # [1, 3, 1024, 768]

# Remove preprocessing padding, resize to original size, and apply focal-length scale
original_size = (image.height, image.width)
result = image_processor.post_process_pointmap_estimation(
    outputs, source_sizes=[original_size], target_sizes=[original_size]
)
pointmap = result[0]["pointmap"]
print("Pointmap shape:", pointmap.shape)  # [3, original_height, original_width]

# Visualize the pointmap as an RGB image using inverse-depth and the turbo colormap.
import matplotlib.pyplot as plt

# `segmentation` is the output of `post_process_semantic_segmentation` — a (H, W) tensor
# of per-pixel class IDs, where class 0 is background.
foreground_mask = segmentation != 0
depth = pointmap[2]  # Z channel: depth in camera space, shape (H, W)
pointmap_rgb = torch.zeros(3, *depth.shape, dtype=torch.uint8)
foreground_depth = depth[foreground_mask]
if foreground_depth.numel() > 0:
    depth_low, depth_high = torch.quantile(foreground_depth, torch.tensor([0.01, 0.99]))
    inverse_depth = 1.0 / foreground_depth.clamp(min=1e-6)
    inverse_depth_low = 1.0 / depth_high.clamp(min=1e-6)
    inverse_depth_high = 1.0 / depth_low.clamp(min=1e-6)
    inverse_depth_normalized = ((inverse_depth - inverse_depth_low) / (inverse_depth_high - inverse_depth_low + 1e-8)).clamp(0, 1)
    turbo = plt.get_cmap("turbo")
    foreground_colors = torch.from_numpy(turbo(inverse_depth_normalized.cpu().numpy())[..., :3] * 255).to(torch.uint8)  # (N, 3)
    pointmap_rgb[:, foreground_mask] = foreground_colors.T
print("Pointmap RGB shape:", pointmap_rgb.shape)  # [3, original_height, original_width]
```

</hfoption>

<hfoption id="Pose estimation">

The example below shows how to run pose estimation with [`Sapiens2ForPoseEstimation`].
The model predicts per-keypoint heatmaps; use `post_process_pose_estimation` to decode them back to
image-space keypoint coordinates. It requires `opencv-python` (`pip install opencv-python`).

```python
import torch
from transformers import AutoImageProcessor, AutoModelForPoseEstimation
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pose-0.4b", revision="refs/pr/1")
model = AutoModelForPoseEstimation.from_pretrained("facebook/sapiens2-pose-0.4b", device_map="auto", revision="refs/pr/1")

# Provide bounding boxes in COCO format (x, y, width, height) for each person
boxes = [[[270.8, 0.6, 294.1, 379.5]]]
inputs = image_processor(image, boxes=boxes, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.heatmaps shape: (num_persons, num_keypoints, heatmap_height, heatmap_width)
print("Heatmaps shape:", outputs.heatmaps.shape)  # [1, 308, 256, 192]

# Decode heatmaps to image-space keypoint coordinates
results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)[0]
keypoints = results[0]["keypoints"]   # (num_keypoints, 2) — x/y in image coordinates
scores = results[0]["scores"]         # (num_keypoints,) — per-keypoint confidence
print("Keypoints shape:", keypoints.shape)
```

</hfoption>

<hfoption id="Pose estimation with flip augmentation">

Horizontal flip augmentation (test-time augmentation) improves keypoint accuracy by averaging
predictions from the original and mirrored image. Pass `flip_pairs` — a tensor of
`[left_keypoint, right_keypoint]` pairs — to the second forward pass. The model flips the heatmaps
back to the original orientation before returning them, so you can average both outputs directly.

```python
import torch
from transformers import AutoImageProcessor, AutoModelForPoseEstimation
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-pose-0.4b", revision="refs/pr/1")
model = AutoModelForPoseEstimation.from_pretrained("facebook/sapiens2-pose-0.4b", device_map="auto", revision="refs/pr/1")

boxes = [[[270.8, 0.6, 294.1, 379.5]]]
inputs = image_processor(image, boxes=boxes, return_tensors="pt").to(model.device)
pixel_values = inputs["pixel_values"]

flip_pairs = torch.tensor(model.config.flip_pairs, device=model.device)

with torch.inference_mode():
    outputs = model(pixel_values)
    outputs_flipped = model(pixel_values.flip(-1), flip_pairs=flip_pairs)

results = image_processor.post_process_pose_estimation(outputs, outputs_flipped=outputs_flipped, boxes=boxes)[0]
keypoints = results[0]["keypoints"]
scores = results[0]["scores"]
```

</hfoption>

<hfoption id="Semantic segmentation">

The example below shows how to perform body-part segmentation with [`Sapiens2ForSemanticSegmentation`].

```python
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-seg-0.4b", revision="refs/pr/1")
model = AutoModelForSemanticSegmentation.from_pretrained("facebook/sapiens2-seg-0.4b", device_map="auto", revision="refs/pr/1")

inputs = image_processor(image, return_tensors="pt").to(model.device)
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

</hfoption>

<hfoption id="Matting">

The example below shows how to run image matting with [`Sapiens2ForImageMatting`].
Outputs are sigmoid-activated and already in `[0, 1]`; use `post_process_image_matting` to resize and split
into `alphas`, `foregrounds`, and an optional `composite` image. The composite image shows
the foreground overlaid over the background with the formula: `composite = foreground * (1 - alpha) * background`.

```python
import torch
from transformers import AutoImageProcessor, AutoModelForImageMatting
from transformers.image_utils import load_image

image = load_image("http://images.cocodataset.org/val2017/000000004016.jpg")

image_processor = AutoImageProcessor.from_pretrained("facebook/sapiens2-matting-1b", revision="refs/pr/1")
model = AutoModelForImageMatting.from_pretrained("facebook/sapiens2-matting-1b", device_map="auto", revision="refs/pr/1")

inputs = image_processor(image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

# outputs.foregrounds: (1, 3, H, W), outputs.alphas: (1, 1, H, W) — both in [0, 1]
original_size = (image.height, image.width)

# Pass an optional background to composite the foreground over it.
# A (3, 1, 1) tensor broadcasts as a uniform color; PIL images and numpy arrays are also accepted.
background = torch.tensor([0, 177, 64], dtype=torch.uint8).view(3, 1, 1)  # chroma green in RGB
result = image_processor.post_process_image_matting(
    outputs, target_sizes=[original_size], backgrounds=background
)[0]
print("Alpha shape:", result["alpha"].shape)        # [1, original_height, original_width]
print("Foreground shape:", result["foreground"].shape)  # [3, original_height, original_width]
print("Composite shape:", result["composite"].shape)    # [3, original_height, original_width] — uint8 [0, 255]
```

</hfoption>

</hfoptions>

## Sapiens2Config

[[autodoc]] Sapiens2Config

## Sapiens2HeadConfig

[[autodoc]] Sapiens2HeadConfig

## Sapiens2ImageProcessor

[[autodoc]] Sapiens2ImageProcessor
    - preprocess
    - post_process_image_matting
    - post_process_normal_estimation
    - post_process_pointmap_estimation
    - post_process_pose_estimation
    - post_process_semantic_segmentation

## Sapiens2Model

[[autodoc]] Sapiens2Model
    - forward

## Sapiens2Backbone

[[autodoc]] Sapiens2Backbone
    - forward

## Sapiens2ForImageMatting

[[autodoc]] Sapiens2ForImageMatting
    - forward

## Sapiens2ForNormalEstimation

[[autodoc]] Sapiens2ForNormalEstimation
    - forward

## Sapiens2ForPointmapEstimation

[[autodoc]] Sapiens2ForPointmapEstimation
    - forward

## Sapiens2ForPoseEstimation

[[autodoc]] Sapiens2ForPoseEstimation
    - forward

## Sapiens2ForSemanticSegmentation

[[autodoc]] Sapiens2ForSemanticSegmentation
    - forward