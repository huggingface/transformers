<!--Copyright 2025 The HuggingFace Team. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. ⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer. -->

This model was released in 2025 and added to Hugging Face Transformers in 2025-10. [web:28][web:25]

DEIMv2
<div style="float: right;"> <div class="flex flex-wrap space-x-1"> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white"> <img alt="Object Detection" src="https://img.shields.io/badge/Object%20Detection-0ea5e9?style=flat"> <img alt="AutoBackbone" src="https://img.shields.io/badge/AutoBackbone-16a34a?style=flat"> </div> </div>
Overview

DEIMv2 is a real‑time object detection architecture built on DINOv3 features, introducing a Spatial Tuning Adapter (STA) to convert single‑scale ViT features into a lightweight multi‑scale pyramid, a simplified decoder, and an upgraded Dense one‑to‑one matching strategy. [web:16][web:6]

This integration uses the AutoBackbone API so DINO‑family backbones can be reused without re‑implementation in the detection head; the initial release targets DINOv3/ViT backbones, with tiny HGNetv2 variants planned as follow‑ups.

[!TIP]
The smallest working example below shows how to run inference and obtain boxes, scores, and labels from post‑processing. [web:25][web:28]

<hfoptions id="usage"> <hfoption id="Pipeline">

from PIL import Image
from transformers import pipeline

detector = pipeline(
    task="object-detection",
    model="your-org/deimv2-dinov3-base"
)
image = Image.open("path/to/your/image.jpg")
outputs = detector(image)
print(outputs[:3])

</hfoption> <hfoption id="AutoModel">

from PIL import Image
import requests
from transformers import Deimv2ImageProcessor, Deimv2ForObjectDetection

ckpt = "your-org/deimv2-dinov3-base" # replace when a checkpoint is available
model = Deimv2ForObjectDetection.from_pretrained(ckpt)
processor = Deimv2ImageProcessor.from_pretrained(ckpt)

url = "https://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor.preprocess([image], return_tensors="pt")
outputs = model(**inputs)
results = processor.post_process_object_detection(outputs, threshold=0.5)
print(results)

</hfoption> <hfoption id="transformers CLI">

echo -e "https://images.cocodataset.org/val2017/000000039769.jpg" | transformers run \
--task object-detection \
--model your-org/deimv2-dinov3-base

</hfoption> </hfoptions>
Model notes

Backbone via AutoBackbone: loads DINOv3/ViT variants and exposes feature maps to the DEIMv2 head.

Spatial Tuning Adapter (STA): transforms single‑scale features into a multi‑scale pyramid for accurate localization with minimal overhead.

Decoder and Dense O2O: streamlined decoder with one‑to‑one assignment for stable training and real‑time throughput.

Expected inputs and outputs

Inputs: pixel_values shaped 
𝐵
×
3
×
𝐻
×
𝑊
B×3×H×W, produced by Deimv2ImageProcessor.preprocess.

Outputs: class logits 
𝐵
×
𝑄
×
𝐶
B×Q×C and normalized pred_boxes 
𝐵
×
𝑄
×
4
B×Q×4; use post_process_object_detection to filter and convert to absolute coordinates.

Configuration

[[autodoc]] Deimv2Config

__init__

This configuration defines backbone settings, query count, decoder depth, STA parameters, and sets model_type="deimv2". Any changes to the configuration (e.g., hidden_dim, num_queries, or STA scale factors) are reflected in model initialization.

Base model

[[autodoc]] Deimv2Model

forward

Connects the backbone to STA and decoder. Returns decoder hidden states for the detection head.

Task head

[[autodoc]] Deimv2ForObjectDetection

forward

Predicts class logits and normalized bounding boxes for a fixed set of queries. Compatible with the post-processing API to get final detection outputs.

Image Processor

[[autodoc]] Deimv2ImageProcessor

preprocess

post_process_object_detection

Handles resizing, normalization, batching, and conversion of model outputs to boxes, scores, and labels. Supports different input image sizes and batch processing.

Resources

Paper: “Real‑Time Object Detection Meets DINOv3.” [web:16][web:7]

Official repository and model zoo for reference implementations and weights. [web:3][web:12]

AutoBackbone documentation for reusing vision backbones. [web:17][web:28]

Citations

Please cite the original DEIMv2 paper when using this model: “Real‑Time Object Detection Meets DINOv3.” [web:16][web:7]