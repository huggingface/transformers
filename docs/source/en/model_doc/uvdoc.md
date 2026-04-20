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
*This model was released on 2023-02-06 and added to Hugging Face Transformers on 2026-03-21.*

# UVDoc

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**UVDoc** The main purpose of text image correction is to carry out geometric transformation on the image to correct the document distortion, inclination, perspective deformation and other problems in the image.

## Usage

### Single input inference

The example below demonstrates how to rectify a document image with UVDoc using the [`AutoImageProcessor`] and [`UVDocModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

model_path = "PaddlePaddle/UVDoc_safetensors"
model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/doc_test.jpg", stream=True).raw)

inputs = image_processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)

result = image_processor.post_process_document_rectification(outputs.last_hidden_state, inputs["original_images"])
print(result)
```

</hfoption>
</hfoptions>

### Batched inference

Here is how to perform batched document rectification with UVDoc:

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

model_path = "PaddlePaddle/UVDoc_safetensors"
model = AutoModel.from_pretrained(
    model_path
    device_map="auto",
)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/doc_test.jpg", stream=True).raw)

inputs = image_processor(images=[image, image], return_tensors="pt").to(model.device)
outputs = model(**inputs)

result = image_processor.post_process_document_rectification(outputs.last_hidden_state, inputs["original_images"])
print(result)
```

</hfoption>
</hfoptions>

## UVDocConfig

[[autodoc]] UVDocConfig

## UVDocModel

[[autodoc]] UVDocModel

## UVDocBackboneConfig

[[autodoc]] UVDocBackboneConfig

## UVDocBackbone

[[autodoc]] UVDocBackbone

## UVDocBridge

[[autodoc]] UVDocBridge

## UVDocImageProcessor

[[autodoc]] UVDocImageProcessor
