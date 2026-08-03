<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-05-19.*

# PP-OCRv6_medium_det

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

PP-OCRv6_medium_det is the largest model in the PP-OCRv6 detection series developed by the PaddleOCR team. It uses LCNetV4 as the backbone and RepLKFPN as the feature pyramid neck, providing accurate text localization across diverse scenarios including handwritten, printed, rotated, curved, and artistic text in multiple languages. The model contains 15.5M parameters.

## Model Architecture

<img src="https://cdn-uploads.huggingface.co/production/uploads/684ba591e717a30275a1b76a/ofnSGExgJL6K6d8ghh0vl.png" width="600"/>

## Usage

### Single input inference

The example below demonstrates how to detect text with PP-OCRv6_medium_det using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
from io import BytesIO

import httpx
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import load_image

model_path = "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png"
image = load_image(image_url)
inputs = image_processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs, 
    target_sizes=inputs["target_sizes"],
    threshold=0.2,
    box_threshold=0.45,
    max_candidates=3000,
    unclip_ratio=1.4,
)

for result in results:
    print(result)
```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-OCRv6_medium_det using the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
from io import BytesIO

import httpx
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import load_image

model_path = "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png"
image = load_image(image_url)
inputs = image_processor(images=[image, image], return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs, 
    target_sizes=inputs["target_sizes"],
    threshold=0.2,
    box_threshold=0.45,
    max_candidates=3000,
    unclip_ratio=1.4,
)

for result in results:
    print(result)
```

</hfoption>
</hfoptions>

## PPOCRV6MediumDetForObjectDetection

[[autodoc]] PPOCRV6MediumDetForObjectDetection

## PPOCRV6MediumDetConfig

[[autodoc]] PPOCRV6MediumDetConfig

## PPOCRV6MediumDetModel

[[autodoc]] PPOCRV6MediumDetModel

