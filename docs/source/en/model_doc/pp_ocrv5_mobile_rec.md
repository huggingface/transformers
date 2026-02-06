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

# PP-OCRv5_mobile_rec

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**PP-OCRv5_mobile_rec** is a dedicated lightweight model for text recognition, focusing specifically on efficient recognition and understanding of text elements in multi-language documents and natural scenes.

## Model Architecture

PP-OCRv5_mobile_rec is one of the PP-OCRv5_rec series, the latest generation of text recognition models developed by the PaddleOCR team. It is designed to efficiently and accurately support the recognition of Simplified Chinese, Traditional Chinese, English, Japanese, as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters with a single model. While maintaining recognition performance, it also balances inference speed and model robustness, providing efficient and accurate technical support for document understanding in various scenarios. 


## Usage

### Single input inference

The example below demonstrates how to detect text with PP-OCRv5_mobile_rec using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

model_path="PaddlePaddle/PP-OCRv5_mobile_rec_safetensors"
model = AutoModel.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png", stream=True).raw).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt")['pixel_values']
outputs = model(inputs)

results = image_processor.post_process_text_recognition(outputs)

for result in results:
    print(result)

```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-OCRv5_mobile_rec using the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

model_path="PaddlePaddle/PP-OCRv5_mobile_rec_safetensors"
model = AutoModel.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png", stream=True).raw).convert("RGB")
inputs = image_processor(images=[image, image], return_tensors="pt")['pixel_values']
outputs = model(inputs)

results = image_processor.post_process_text_recognition(outputs)

for result in results:
    print(result)

```

</hfoption>
</hfoptions>

## PPOCRV5MobileRecForTextRecognition

[[autodoc]] PPOCRV5MobileRecForTextRecognition

## PPOCRV5MobileRecConfig

[[autodoc]] PPOCRV5MobileRecConfig

## PPOCRV5MobileRecModel

[[autodoc]] PPOCRV5MobileRecModel

## PPOCRV5MobileRecImageProcessor

[[autodoc]] PPOCRV5MobileRecImageProcessor

## PPOCRV5MobileRecImageProcessorFast

[[autodoc]] PPOCRV5MobileRecImageProcessorFast

## PPOCRV5MobileRecImageProcessor

[[autodoc]] PPOCRV5MobileRecImageProcessor