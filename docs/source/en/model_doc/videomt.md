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
*This model was released on 2026-02-19 and added to Hugging Face Transformers on 2026-03-13.*


# VidEoMT

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The VidEoMT model was proposed in [Your ViT is Secretly Also a Video Segmentation Model](https://huggingface.co/papers/2602.17807) by Narges Norouzi, Idil Esen Zulfikar, Niccolò Cavagnero, Tommie Kerssies, Bastian Leibe, Gijs Dubbelman, Daan de Geus. Video Encoder-only Mask Transformer (VidEoMT) is a lightweight encoder-only model for online video segmentation built on a plain [Vision Transformer (ViT)](vit). It is a minimal extension of [EoMT](./eomt) to video which performs both spatial and temporal reasoning within the ViT encoder, without relying on dedicated tracking modules or heavy task-specific heads.

The abstract from the paper is the following:

*Existing online video segmentation models typically combine a per-frame segmenter with complex specialized tracking modules. While effective, these modules introduce significant architectural complexity and computational overhead. Recent studies suggest that plain Vision Transformer (ViT) encoders, when scaled with sufficient capacity and large-scale pre-training, can conduct accurate image segmentation without requiring specialized modules. Motivated by this observation, we propose the Video Encoder-only Mask Transformer (VidEoMT), a simple encoder-only video segmentation model that eliminates the need for dedicated tracking modules. To enable temporal modeling in an encoder-only ViT, VidEoMT introduces a lightweight query propagation mechanism that carries information across frames by reusing queries from the previous frame. To balance this with adaptability to new content, it employs a query fusion strategy that combines the propagated queries with a set of temporally-agnostic learned queries. As a result, VidEoMT attains the benefits of a tracker without added complexity, achieving competitive accuracy while being 5x--10x faster, running at up to 160 FPS with a ViT-L backbone.*

Tips:

- VidEoMT currently only supports a DINOv2 backbone (with register tokens). Available model sizes are ViT-S, ViT-B, and ViT-L.
- The model accepts video input as a 5D tensor of shape `(batch_size, num_frames, 3, height, width)`.
- VidEoMT supports three video segmentation tasks: **instance**, **semantic**, and **panoptic** segmentation, each with a dedicated post-processing method on the video processor.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/tue-mps/videomt).

## Architecture Info

VidEoMT builds on [EoMT](./eomt), which repurposes a plain DINOv2-pretrained Vision Transformer with **register tokens** as a segmentation model. EoMT introduces learned **object queries** and a lightweight **mask prediction head** directly inside the ViT encoder, eliminating the need for task-specific decoders.

VidEoMT extends this to video with two key additions:

1. **Query propagation**: object queries from the previous frame are carried forward to the next frame through a linear projection (`query_updater`), enabling temporal reasoning without a dedicated tracker.
2. **Query fusion**: the propagated queries are added to a set of temporally-agnostic learned queries, allowing the model to adapt to new objects appearing in the video.

The early encoder layers process all frames independently (in parallel), while the final blocks operate per-frame with the fused queries, producing per-frame mask and class predictions.

## Usage Examples

Use the Hugging Face implementation of VidEoMT for inference with pre-trained models. The examples below reuse the public `tue-mps/videomt-dinov2-small-ytvis2019` checkpoint to demonstrate video instance, semantic, and panoptic post-processing on a sample video.

### Video Instance Segmentation

```python
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import AutoModelForUniversalSegmentation, AutoVideoProcessor
from transformers.video_utils import load_video


model_id = "tue-mps/videomt-dinov2-small-ytvis2019"
processor = AutoVideoProcessor.from_pretrained(model_id)
model = AutoModelForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

video_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/videos/pexels-allan-mas-5362370.mp4"
# Sample 8 frames to keep the example lightweight.
video_frames, _ = load_video(video_url, num_frames=8)

inputs = processor(videos=[video_frames], return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

original_height, original_width = video_frames[0].shape[:2]
target_sizes = [(original_height, original_width)] * len(video_frames)

results = processor.post_process_instance_segmentation(
    outputs,
    target_sizes=target_sizes,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, (ax, frame, result) in enumerate(zip(axes.flatten(), video_frames, results)):
    ax.imshow(frame)
    seg = result["segmentation"].cpu().numpy()
    masked = np.ma.masked_where(seg == -1, seg)
    ax.imshow(masked, alpha=0.6, cmap="tab20")
    ax.set_title(f"Frame {idx}")
    ax.axis("off")
plt.suptitle("Video Instance Segmentation")
plt.tight_layout()
plt.show()
```

### Video Semantic Segmentation

```python
import matplotlib.pyplot as plt
import torch

from transformers import AutoModelForUniversalSegmentation, AutoVideoProcessor
from transformers.video_utils import load_video


model_id = "tue-mps/videomt-dinov2-small-ytvis2019"
processor = AutoVideoProcessor.from_pretrained(model_id)
model = AutoModelForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

video_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/videos/pexels-allan-mas-5362370.mp4"
# Sample 8 frames to keep the example lightweight.
video_frames, _ = load_video(video_url, num_frames=8)

inputs = processor(videos=[video_frames], return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

original_height, original_width = video_frames[0].shape[:2]
target_sizes = [(original_height, original_width)] * len(video_frames)

preds = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, (ax, frame, seg_map) in enumerate(zip(axes.flatten(), video_frames, preds)):
    ax.imshow(frame)
    ax.imshow(seg_map.cpu().numpy(), alpha=0.6, cmap="tab20")
    ax.set_title(f"Frame {idx}")
    ax.axis("off")
plt.suptitle("Video Semantic Segmentation")
plt.tight_layout()
plt.show()
```

### Video Panoptic Segmentation

```python
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import AutoModelForUniversalSegmentation, AutoVideoProcessor
from transformers.video_utils import load_video


model_id = "tue-mps/videomt-dinov2-small-ytvis2019"
processor = AutoVideoProcessor.from_pretrained(model_id)
model = AutoModelForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

video_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/videos/pexels-allan-mas-5362370.mp4"
# Sample 8 frames to keep the example lightweight.
video_frames, _ = load_video(video_url, num_frames=8)

inputs = processor(videos=[video_frames], return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

original_height, original_width = video_frames[0].shape[:2]
target_sizes = [(original_height, original_width)] * len(video_frames)

results = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, (ax, frame, result) in enumerate(zip(axes.flatten(), video_frames, results)):
    ax.imshow(frame)
    seg = result["segmentation"].cpu().numpy()
    masked = np.ma.masked_where(seg == -1, seg)
    ax.imshow(masked, alpha=0.6, cmap="tab20")
    ax.set_title(f"Frame {idx}")
    ax.axis("off")
plt.suptitle("Video Panoptic Segmentation")
plt.tight_layout()
plt.show()
```

## VideomtVideoProcessor

[[autodoc]] VideomtVideoProcessor
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## VideomtConfig

[[autodoc]] VideomtConfig

## VideomtPreTrainedModel

[[autodoc]] VideomtPreTrainedModel
    - forward

## VideomtForUniversalSegmentation

[[autodoc]] VideomtForUniversalSegmentation
