# PP-OCRv5_mobile_det

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**PP-OCRv5_mobile_det** is a dedicated lightweight model for text detection, focusing specifically on efficient detection and understanding of text elements in multi-language documents and natural scenes.

## Model Architecture

PP-OCRv5_mobile_det is one of the PP-OCRv5_det series, the latest generation of text detection models developed by the PaddleOCR team. It aims to efficiently and accurately supports the detection of text in diverse scenarios—including handwriting, vertical, rotated, and curved text—across multiple languages such as Simplified Chinese, Traditional Chinese, English, and Japanese. Key features include robust handling of complex layouts, varying text sizes, and challenging backgrounds, making it suitable for practical applications like document analysis, license plate recognition, and scene text detection. 


## Usage

### Single input inference

The example below demonstrates how to detect text with PP-OCRV5_Mobile_Det using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", stream=True).raw).convert("RGB")
inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=inputs["target_sizes"])

for result in results:
    print(result["boxes"])
    print(result["scores"])

```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-OCRV5_Mobile_Det using the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", stream=True).raw).convert("RGB")
inputs = image_processor(images=[image, image], return_tensors="pt")
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

## PPOCRV5MobileDetImageProcessorFast

[[autodoc]] PPOCRV5MobileDetImageProcessorFast

## PPOCRV5MobileDetImageProcessor

[[autodoc]] PPOCRV5MobileDetImageProcessor
