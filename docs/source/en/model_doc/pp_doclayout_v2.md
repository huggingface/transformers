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
*This model was released on 2025-10-16 and added to Hugging Face Transformers on 2026-01-27.*

# PP-DocLayoutV2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**PP-DocLayoutV2** is a dedicated lightweight model for layout analysis, focusing specifically on element detection, classification, and reading order prediction. 

## Model Architecture

PP-DocLayoutV2 is composed of two sequentially connected networks. The first is an RT-DETR-based detection model that performs layout element detection and classification. The detected bounding boxes and class labels are then passed to a subsequent pointer network, which is responsible for ordering these layout elements.

<div align="center">
<img src="https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/PP-DocLayoutV2.png" width="800"/>
</div>

## Usage

### Single input inference

The example below demonstrates how to generate text with PP-DocLayoutV2 using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from PIL import Image
from transformers import pipeline

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
layout_detector = pipeline("object-detection", model="PaddlePaddle/PP-DocLayoutV2_safetensors")
result = layout_detector(image)
print(result)
```

</hfoption>

<hfoption id="AutoModel">

```py
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_path = "PaddlePaddle/PP-DocLayoutV2_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)
image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]))
for result in results:
    print(result["scores"])
    print(result["labels"])
    print(result["boxes"])
    for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {idx + 1}: {model.config.id2label[label]}: {score:.2f} {box}")
```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-DocLayoutV2 using [`Pipeline`] or the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from PIL import Image
from transformers import pipeline

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
layout_detector = pipeline("object-detection", model="PaddlePaddle/PP-DocLayoutV2_safetensors")
result = layout_detector([image, image])
print(result[0])
print(result[1])
```

</hfoption>

<hfoption id="AutoModel">

```py
from transformers import AutoImageProcessor, AutoModelForObjectDetection

model_path = "PaddlePaddle/PP-DocLayoutV2_safetensors"
model = AutoModelForObjectDetection.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg", stream=True).raw)
inputs = image_processor(images=[image, image], return_tensors="pt")
target_sizes = [image.size[::-1], image.size[::-1]]

outputs = model(**inputs)
results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes)
for result in results:
    print("result:")
    for idx, (score, label_id, box) in enumerate(zip(result["scores"], result["labels"], result["boxes"])):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"Order {idx + 1}: {model.config.id2label[label]}: {score:.2f} {box}")
```

</hfoption>
</hfoptions>

## PPDocLayoutV2ForObjectDetection

[[autodoc]] PPDocLayoutV2ForObjectDetection

## PPDocLayoutV2Config

[[autodoc]] PPDocLayoutV2Config

## PPDocLayoutV2Model

[[autodoc]] PPDocLayoutV2Model

## PPDocLayoutV2ImageProcessorFast

[[autodoc]] PPDocLayoutV2ImageProcessorFast

## PPDocLayoutV2ImageProcessor

[[autodoc]] PPDocLayoutV2ImageProcessor
