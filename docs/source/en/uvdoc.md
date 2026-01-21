# UVDoc

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**UVDoc** The main purpose of text image correction is to carry out geometric transformation on the image to correct the document distortion, inclination, perspective deformation and other problems in the image.

## Model Architecture 
The main purpose of text image correction is to carry out geometric transformation on the image to correct the document distortion, inclination, perspective deformation and other problems in the image.


## Usage

### Single input inference

The example below demonstrates how to classify image with UVDoc using the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, UVDocForDocumentRectification

model_path = "PaddlePaddle/UVDoc_safetensors"
model = UVDocForDocumentRectification.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/doc_test.jpg", stream=True).raw)

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

result = image_processor.post_process_document_rectification(outputs.logits)
print(result)
```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-LCNet using [`Pipeline`] or the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, UVDocForDocumentRectification

model_path = "PaddlePaddle/UVDoc_safetensors"
model = UVDocForDocumentRectification.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/doc_test.jpg", stream=True).raw)

inputs = image_processor(images=[image, image], return_tensors="pt")
outputs = model(**inputs)

result = image_processor.post_process_document_rectification(outputs.logits)
print(result)
```

</hfoption>
</hfoptions>

## UVDocForDocumentRectification

[[autodoc]] UVDocForDocumentRectification

## UVDocConfig

[[autodoc]] UVDocConfig

## UVDocModel

[[autodoc]] UVDocModel

## UVDocImageProcessorFast

[[autodoc]] UVDocImageProcessorFast

## UVDocImageProcessor

[[autodoc]] UVDocImageProcessor
