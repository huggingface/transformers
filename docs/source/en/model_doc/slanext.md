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
*This model was released on 2025-03-07 and added to Hugging Face Transformers on 2026-03-21.*

# SLANeXt

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**SLANeXt_wired** and **SLANeXt_wireless** are part of a series of dedicated lightweight models for table structure recognition, focusing on accurately recognizing table structures in documents and natural scenes. For more details about the SLANeXt series model, please refer to the [official documentation](https://www.paddleocr.ai/latest/en/version3.x/module_usage/table_structure_recognition.html).

## Model Architecture

The SLANeXt series is a new generation of table structure recognition models independently developed by the Baidu PaddlePaddle Vision Team. SLANeXt focuses on table structure recognition, and trains dedicated weights for wired and wireless tables separately. The recognition ability for all types of tables has been significantly improved, especially for wired tables.


## Usage

### Single input inference

The example below demonstrates how to detect text with PP-OCRV5_Mobile_Det using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForTableRecognition

model_path="PaddlePaddle/SLANeXt_wired_safetensors"
model = AutoModelForTableRecognition.from_pretrained(model_path, dtype=torch.float32, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg", stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_table_recognition(outputs)

print(result['structure'])
print(result['structure_score'])
```

</hfoption>
</hfoptions>

## SLANeXtConfig

[[autodoc]] SLANeXtConfig

## SLANeXtForTableRecognition

[[autodoc]] SLANeXtForTableRecognition

## SLANeXtBackbone

[[autodoc]] SLANeXtBackbone

## SLANeXtSLAHead

[[autodoc]] SLANeXtSLAHead

## SLANeXtImageProcessor

[[autodoc]] SLANeXtImageProcessor
