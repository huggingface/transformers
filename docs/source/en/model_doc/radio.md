<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-30.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# RADIO

[RADIO](https://huggingface.co/papers/2312.06709) (Reduce All Domains Into One) is a family of vision foundation models from NVIDIA trained by multi-teacher distillation (e.g. CLIP, DINOv2, SAM) into a single ViT backbone. It produces both an image-level `summary` embedding and dense spatial `features`, and supports variable input resolutions through a Cropped Position Embedding (CPE) patch generator.

The example below demonstrates how to extract image features with the [`RadioModel`] class.

<hfoptions id="usage">
<hfoption id="RadioModel">

```python
import requests
import torch
from PIL import Image

from transformers import CLIPImageProcessor, RadioModel


hf_repo = "nvidia/C-RADIOv4-H"

model = RadioModel.from_pretrained(hf_repo)
model.eval().cuda()

image_processor = CLIPImageProcessor(
    size={"height": 224, "width": 224}, do_resize=True, do_center_crop=False, do_normalize=False
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
pixel_values = pixel_values.cuda()

with torch.no_grad():
    outputs = model(pixel_values)

summary = outputs.summary    # (1, 2560) image-level embedding
features = outputs.features   # (1, 196, 1280) dense spatial features
```

</hfoption>
</hfoptions>

## RadioConfig

[[autodoc]] RadioConfig

## RadioModel

[[autodoc]] RadioModel
    - forward
