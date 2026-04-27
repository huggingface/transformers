<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-09-25 and added to Hugging Face Transformers on 2026-04-27.*

# DEIMv2

## Overview

DEIMv2 (DETR with Improved Matching v2) was proposed in [DEIMv2: Real-Time Object Detection Meets DINOv3](https://huggingface.co/papers/2509.20787) by Shihua Huang, Yongjie Hou, Longfei Liu, Xuanlong Yu, and Xi Shen.

The abstract from the paper is the following:

*Driven by the simple and effective Dense O2O, DEIM demonstrates faster convergence and enhanced performance. In this work, we extend it with DINOv3 features, resulting in DEIMv2. DEIMv2 spans eight model sizes from X to Atto, covering GPU, edge, and mobile deployment. For the X, L, M, and S variants, we adopt DINOv3-pretrained / distilled backbones and introduce a Spatial Tuning Adapter (STA), which efficiently converts DINOv3's single-scale output into multi-scale features and complements strong semantics with fine-grained details to enhance detection. For ultra-lightweight models (Nano, Pico, Femto, and Atto), we employ HGNetv2 with depth and width pruning to meet strict resource budgets. Together with a simplified decoder and an upgraded Dense O2O, this unified design enables DEIMv2 to achieve a superior performance-cost trade-off across diverse scenarios, establishing new state-of-the-art results. Notably, our largest model, DEIMv2-X, achieves 57.8 AP with only 50.3M parameters, surpassing prior X-scale models that require over 60M parameters for just 56.5 AP. On the compact side, DEIMv2-S is the first sub-10M model (9.71M) to exceed the 50 AP milestone on COCO, reaching 50.9 AP. Even the ultra-lightweight DEIMv2-Pico, with just 1.5M parameters, delivers 38.5 AP-matching YOLOv10-Nano (2.3M) with ~50% fewer parameters.*

## Usage

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

image_processor = AutoImageProcessor.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers")
model = AutoModelForObjectDetection.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers", device_map="auto")

inputs = image_processor(images=image, return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=[image.size[::-1]]
)

for result in results:
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
```

## Deimv2Config

[[autodoc]] Deimv2Config

## Deimv2Model

[[autodoc]] Deimv2Model
    - forward

## Deimv2ForObjectDetection

[[autodoc]] Deimv2ForObjectDetection
    - forward
