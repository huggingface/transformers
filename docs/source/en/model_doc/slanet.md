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
*This model was released on 2025-03-07 and added to Hugging Face Transformers on 2026-04-22.*

# SLANet


## Overview

**SLANet** and **SLANet_plus** are part of a series of dedicated lightweight models for table structure recognition, focusing on accurately recognizing table structures in documents and natural scenes. For more details about the SLANet series model, please refer to the [official documentation](https://www.paddleocr.ai/latest/en/version3.x/module_usage/table_structure_recognition.html).

## Model Architecture

SLANet is a table structure recognition model developed by Baidu PaddlePaddle Vision Team. The model significantly improves the accuracy and inference speed of table structure recognition by adopting a CPU-friendly lightweight backbone network PP-LCNet, a high-low-level feature fusion module CSP-PAN, and a feature decoding module SLA Head that aligns structural and positional information.

## Usage

### Single input inference

The example below demonstrates how to detect text with SLANet using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
from io import BytesIO

import httpx
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForTableRecognition


model_path="PaddlePaddle/SLANet_plus_safetensors"
model = AutoModelForTableRecognition.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(BytesIO(httpx.get(image_url).content))
inputs = image_processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_table_recognition(outputs)

print(result['structure'])
print(result['structure_score'])
```

</hfoption>
</hfoptions>

## SLANetConfig

[[autodoc]] SLANetConfig

## SLANetForTableRecognition

[[autodoc]] SLANetForTableRecognition

## SLANetBackbone

[[autodoc]] SLANetBackbone

## SLANetSLAHead

[[autodoc]] SLANetSLAHead

