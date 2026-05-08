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
*This model was released on 2026-05-08 and added to Hugging Face Transformers on 2026-05-08.*

# PP-OCRv6_base_rec

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

TODO.

## Model Architecture

TODO.

## Usage

### Single input inference

The example below demonstrates how to detect text with PP-OCRv6_base_rec using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
from io import BytesIO

import httpx
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForTextRecognition

model_path = "PaddlePaddle/PP-OCRv6_base_rec_safetensors"
model = AutoModelForTextRecognition.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png"
image = Image.open(BytesIO(httpx.get(image_url).content))
inputs = image_processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_text_recognition(outputs)
for result in results:
    print(result)
```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-OCRv6_base_rec using the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="AutoModel">

```python
from io import BytesIO

import httpx
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForTextRecognition

model_path = "PaddlePaddle/PP-OCRv6_base_rec_safetensors"
model = AutoModelForTextRecognition.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png"
image = Image.open(BytesIO(httpx.get(image_url).content))
inputs = image_processor(images=[image, image], return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_text_recognition(outputs)
for result in results:
    print(result)
```

</hfoption>
</hfoptions>

## PPOCRV6BaseRecForTextRecognition

[[autodoc]] PPOCRV6BaseRecForTextRecognition

## PPOCRV6BaseRecConfig

[[autodoc]] PPOCRV6BaseRecConfig

## PPOCRV6BaseRecModel

[[autodoc]] PPOCRV6BaseRecModel

## PPOCRV6BaseRecEncoderWithSVTR

[[autodoc]] PPOCRV6BaseRecEncoderWithSVTR

## PPOCRV6BaseRecImageProcessor

[[autodoc]] PPOCRV6BaseRecImageProcessor
