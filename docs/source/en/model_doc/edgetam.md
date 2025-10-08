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

*This model was released on 2025-01-13 and added to Hugging Face Transformers on 2025-10-07 and contributed by [yonigozlan](https://huggingface.co/yonigozlan).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# EdgeTAM

[EdgeTAM: On-Device Track Anything Model](https://huggingface.co/papers/2501.07256) extends SAM 2 for efficient real-time video segmentation on mobile devices by introducing a 2D Spatial Perceiver. This architecture optimizes memory attention mechanisms, addressing latency issues caused by memory attention blocks. The 2D Spatial Perceiver uses a lightweight Transformer with fixed learnable queries, split into global and patch-level groups to preserve spatial structure. Additionally, a distillation pipeline enhances performance without increasing inference time. EdgeTAM achieves high J&F scores on DAVIS 2017, MOSE, SA-V val, and SA-V test, while operating at 16 FPS on iPhone 15 Pro Max.

<hfoptions id="usage">
<hfoption id="EdgeTamModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, EdgeTamModel

model = EdgeTamModel.from_pretrained("yonigozlan/edgetam-1", dtype="auto")
processor = AutoProcessor.from_pretrained("yonigozlan/edgetam-1")

image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

input_points = [[[[500, 375]]]]
input_labels = [[[1]]]

inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")
print(f"IoU scores: {outputs.iou_scores.squeeze()}")
```

</hfoption>
</hfoptions>

## EdgeTamConfig

[[autodoc]] EdgeTamConfig

## EdgeTamVisionConfig

[[autodoc]] EdgeTamVisionConfig

## EdgeTamMaskDecoderConfig

[[autodoc]] EdgeTamMaskDecoderConfig

## EdgeTamPromptEncoderConfig

[[autodoc]] EdgeTamPromptEncoderConfig

## EdgeTamVisionModel

[[autodoc]] EdgeTamVisionModel
    - forward

## EdgeTamModel

[[autodoc]] EdgeTamModel
    - forward

