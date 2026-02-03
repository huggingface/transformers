# SLANeXt

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**SLANeXt_wired** and **SLANeXt_wireless** are part of a series of dedicated lightweight models for table structure recognition, focusing on accurately recognizing table structures in documents and natural scenes.

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
from transformers import AutoImageProcessor, AutoModel

model_path="PaddlePaddle/SLANeXt_wired_safetensors"
model = AutoModel.from_pretrained(model_path).float()
image_processor = AutoModel.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg", stream=True).raw).convert("RGB")
inputs = image_processor(images=image)
outputs = model(inputs)

results = image_processor.post_process_table_recognition(outputs)

print(result['structure'])
print(result['structure_score'])
```

</hfoption>
</hfoptions>


## SLANeXtForTableRecognition

[[autodoc]] SLANeXtForTableRecognition

## SLANeXtConfig

[[autodoc]] SLANeXtConfig

## SLANeXtModel

[[autodoc]] SLANeXtModel

## SLANeXtImageProcessor

[[autodoc]] SLANeXtImageProcessor