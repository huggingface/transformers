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
*This model was released on 2021-09-17 and added to Hugging Face Transformers on 2026-03-11.*

# PP-LCNet

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**PP-LCNet** PP-LCNet is a family of efficient, lightweight convolutional neural networks designed for real-world document understanding and OCR tasks. It balances accuracy, speed, and model size, making it ideal for both server-side and edge deployment. To address different document processing requirements, PP-LCNet has three main variants, each optimized for a specific task.

## Model Architecture

1. The Document Image Orientation Classification Module is primarily designed to distinguish the orientation of document images and correct them through post-processing. During processes such as document scanning or ID photo capturing, the device might be rotated to achieve clearer images, resulting in images with various orientations. Standard OCR pipelines may not handle these images effectively. By leveraging image classification techniques, the orientation of documents or IDs containing text regions can be pre-determined and adjusted, thereby improving the accuracy of OCR processing.

2. The Table Classification Module is a key component in computer vision systems, responsible for classifying input table images. The performance of this module directly affects the accuracy and efficiency of the entire table recognition process. The Table Classification Module typically receives table images as input and, using deep learning algorithms, classifies them into predefined categories based on the characteristics and content of the images, such as wired and wireless tables. The classification results from the Table Classification Module serve as output for use in table recognition pipelines.

3. The text line orientation classification module primarily distinguishes the orientation of text lines and corrects them using post-processing. In processes such as document scanning and license/certificate photography, to capture clearer images, the capture device may be rotated, resulting in text lines in various orientations. Standard OCR pipelines cannot handle such data well. By utilizing image classification technology, the orientation of text lines can be predetermined and adjusted, thereby enhancing the accuracy of OCR processing.


## Usage

### Single input inference

The example below demonstrates how to classify image with PP-LCNet using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from PIL import Image
from transformers import pipeline

model_path = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
image_classifier = pipeline("image-classification", model=model_path, function_to_apply="none", device_map="auto")

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg", stream=True).raw)
result = image_classifier(image)
print(result)
```

</hfoption>

<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_path = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
model = AutoModelForImageClassification.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg", stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_label = outputs.logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

### Batched inference

Here is how you can do it with PP-LCNet using [`Pipeline`] or the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
from PIL import Image
from transformers import pipeline

model_path = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
image_classifier = pipeline("image-classification", model=model_path, function_to_apply="none", device_map="auto")

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg", stream=True).raw)
result = image_classifier([image, image])
print(result)
```

</hfoption>

<hfoption id="AutoModel">

```py
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_path = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
model = AutoModelForImageClassification.from_pretrained(model_path, device_map="auto")
image_processor = AutoImageProcessor.from_pretrained(model_path)

image = Image.open(requests.get("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg", stream=True).raw)
inputs = image_processor(images=[image, image], return_tensors="pt")
outputs = model(**inputs)

predicted_labels = outputs.logits.argmax(-1)

for label_id in predicted_labels:
    label_id_scalar = label_id.item()
    label = model.config.id2label[label_id_scalar]
    print(label)
```

</hfoption>
</hfoptions>

## PPLCNetForImageClassification

[[autodoc]] PPLCNetForImageClassification

## PPLCNetConfig

[[autodoc]] PPLCNetConfig

## PPLCNetModel

[[autodoc]] PPLCNetModel

## PPLCNetBackbone

[[autodoc]] PPLCNetBackbone

## PPLCNetImageProcessorFast

[[autodoc]] PPLCNetImageProcessorFast

## PPLCNetImageProcessor

[[autodoc]] PPLCNetImageProcessor
