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
*This model was released on 2025-05-20 and added to Hugging Face Transformers on 2026-03-13.*

# PP-OCRv5_mobile_det


## Overview

**PP-OCRv5_mobile_det** is a dedicated lightweight model for text detection, focusing specifically on efficient detection and understanding of text elements in multi-language documents and natural scenes.

## Model Architecture

PP-OCRv5_mobile_det is one of the PP-OCRv5_det series, the latest generation of text detection models developed by the PaddleOCR team. It aims to efficiently and accurately supports the detection of text in diverse scenarios—including handwriting, vertical, rotated, and curved text—across multiple languages such as Simplified Chinese, Traditional Chinese, English, and Japanese. Key features include robust handling of complex layouts, varying text sizes, and challenging backgrounds, making it suitable for practical applications like document analysis, license plate recognition, and scene text detection. 


## Usage

### Single input inference

The example below demonstrates how to detect text with PP-OCRV5_Mobile_Det using the [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import requests
from PIL import Image

from transformers import pipeline


image = Image.open(
    requests.get(
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", stream=True
    ).raw)
detector = pipeline(
    task="object-detection",
    model="PaddlePaddle/PP-OCRV5_mobile_det_safetensors",
    device_map="auto",
)
results = detector(image)

for result in results:
    print(result)
```

</hfoption>
<hfoption id="AutoModel">

```python
import requests
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForObjectDetection


model_path="PaddlePaddle/PP-OCRv5_mobile_det_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", stream=True).raw).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=inputs["target_sizes"])

for result in results:
    print(result["boxes"])
    print(result["scores"])
```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-OCRV5_Mobile_Det using the [`Pipeline`] or the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import requests
from PIL import Image

from transformers import pipeline


image = Image.open(
    requests.get(
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", stream=True
    ).raw)
detector = pipeline(
    task="object-detection",
    model="PaddlePaddle/PP-OCRV5_mobile_det_safetensors",
    device_map="auto",
)
results = detector([image, image])

for result in results:
    print(result)
```

</hfoption>

<hfoption id="AutoModel">

```python
import requests
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForObjectDetection


model_path="PaddlePaddle/PP-OCRv5_mobile_det_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", stream=True).raw).convert("RGB")
inputs = image_processor(images=[image, image], return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=inputs["target_sizes"])

for result in results:
    print(result["boxes"])
    print(result["scores"])
```

</hfoption>
</hfoptions>

## PPOCRV5MobileDetForObjectDetection

[[autodoc]] PPOCRV5MobileDetForObjectDetection

## PPOCRV5MobileDetConfig

[[autodoc]] PPOCRV5MobileDetConfig

## PPOCRV5MobileDetModel

[[autodoc]] PPOCRV5MobileDetModel
