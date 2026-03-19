<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DEIMv2

## Overview

DEIMv2 (DETR with Improved Matching v2) was proposed in [DEIMv2: Real-Time Object Detection Meets DINOv3](https://huggingface.co/papers/2509.20787) by Shihua Huang, Yongjie Hou, Longfei Liu, Xuanlong Yu, and Xi Shen.

DEIMv2 builds upon D-FINE's distribution-based bounding box refinement approach, adding several key innovations:
- **SwiGLU FFN**: Replaces the standard MLP in decoder layers with a SwiGLU-gated feed-forward network.
- **RMSNorm**: Uses RMSNorm instead of LayerNorm in decoder layers for improved training stability.
- **RepNCSPELAN5**: An enhanced 5-branch CSP-ELAN encoder block (vs D-Fine's 4-branch RepNCSPELAN4).
- **Matching Auxiliary Loss (MAL)**: A focal-style BCE loss with IoU-weighted targets replacing VFL.
- **Dense O2O Matching**: Unified matching across decoder layers for improved training convergence.

## Usage

```python
from transformers import AutoImageProcessor, Deimv2ForObjectDetection
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

# TODO: Replace with Transformers-compatible ckpts once uploaded.
image_processor = AutoImageProcessor.from_pretrained("Intellindust/DEIMv2_HGNetv2_N_COCO")
model = Deimv2ForObjectDetection.from_pretrained("Intellindust/DEIMv2_HGNetv2_N_COCO")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=[image.size[::-1]]
)

for result in results:
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
```

## Deimv2Config

[[autodoc]] Deimv2Config

## Deimv2Model

[[autodoc]] Deimv2Model
    - forward

## Deimv2ForObjectDetection

[[autodoc]] Deimv2ForObjectDetection
    - forward
