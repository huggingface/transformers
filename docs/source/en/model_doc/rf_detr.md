<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-03-26.* 

# RF-DETR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The RF-DETR model was proposed in the blog post [RF-DETR: A SOTA Real-Time Object Detection Model](https://blog.roboflow.com/rf-detr/) by Peter Robicheaux, James Gallagher, Joseph Nelson, and Isaac Robinson of Roboflow.

RF-DETR ("Roboflow Detection Transformer") is a real-time, transformer-based object detection model designed for high accuracy and strong performance across a wide variety of domains and datasets. It is the first real-time model to exceed 60 AP on the Microsoft COCO benchmark and also achieves state-of-the-art results on the RF100-VL benchmark, which measures a model's ability to adapt to diverse, real-world problems. The architecture combines the principles of LW-DETR with a pre-trained DINOv2 backbone, giving it an exceptional ability to generalize to new domains. Unlike some DETR variants that use multi-scale features, RF-DETR extracts features from a single scale to balance speed and performance.

The model is available in several sizes, from Nano to Large, making it suitable for a range of applications, from high-speed edge devices to tasks requiring maximum precision. It is open-source and released under an Apache 2.0 license.

There is no official paper, but the key motivations are stated in the announcement blog post:

*We are introducing RF-DETR, a state-of-the-art real-time object detection model. RF-DETR outperforms all existing object detection models on real world datasets and is the first real-time model to achieve 60+ mean Average Precision when benchmarked on the COCO dataset. [...] In a world where COCO performance is increasingly saturated for models of similar size, how well a model is able to adapt – especially to novel domains – grows in importance.*

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/roboflow/rf-detr).

## Usage tips

RF-DETR is a powerful choice for real-time object detection, especially when adaptability to new domains is critical.

- **Domain Adaptability** – Thanks to its DINOv2 backbone, RF-DETR is particularly effective at generalizing to new and varied datasets beyond common benchmarks.
- **Multiple Sizes** – The model comes in five variants: Nano, Small, Medium, Base, and Large, allowing users to choose the best trade-off between speed and accuracy for their specific use case.
- **Performance** – RF-DETR is designed for high-speed inference and is competitive with or superior to other real-time models like YOLO, especially when considering the total latency that includes post-processing steps like NMS.
- **Training** – The model can be fine-tuned on custom datasets in COCO format. Roboflow provides extensive documentation and Colab notebooks to guide users through the training process.

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import RfDetrForObjectDetection, AutoImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("stevenbucaille/rf-detr-medium")
>>> model = RfDetrForObjectDetection.from_pretrained("stevenbucaille/rf-detr-medium")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
cat: 0.97[341.14, 25.11, 639.98, 372.89]
cat: 0.96[12.78, 56.35, 317.67, 471.34]
remote: 0.95[39.96, 73.12, 175.65, 117.44]
sofa: 0.86[-0.11, 2.97, 639.89, 473.62]
sofa: 0.82[-0.12, 1.78, 639.87, 473.52]
remote: 0.79[333.65, 76.38, 370.69, 187.48]
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LW Detr.

<PipelineTag pipeline="object-detection"/>

- Scripts for finetuning [`RfDetrForObjectDetection`] with [`Trainer`] or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- See also: [Object detection task guide](../tasks/object_detection).

## RfDetrConfig

[[autodoc]] RfDetrConfig

## RfDetrDinov2WithRegistersConfig

[[autodoc]] RfDetrDinov2WithRegistersConfig

## RfDetrModel

[[autodoc]] RfDetrModel
    - forward

## RfDetrForObjectDetection

[[autodoc]] RfDetrForObjectDetection
    - forward

## RfDetrDinov2WithRegistersBackbone

[[autodoc]] RfDetrDinov2WithRegistersBackbone
    - forward
