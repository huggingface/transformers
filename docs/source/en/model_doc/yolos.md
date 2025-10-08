<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-06-01 and added to Hugging Face Transformers on 2022-05-02 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# YOLOS

[YOLOS](https://huggingface.co/papers/2106.00666) leverages a plain Vision Transformer for object detection, inspired by DETR. It demonstrates that a base-sized encoder-only Transformer can achieve 42 AP on COCO, similar to DETR and more complex frameworks like Faster R-CNN. YOLOS pre-trained on ImageNet-1k achieves competitive performance with minimal modifications and without specific region priors or task-specific inductive biases. The model also explores the impacts and limitations of pre-training schemes and scaling strategies for Transformers in vision tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="hustvl/yolos-base", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")
model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-base", dtype="auto")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
    0
]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
```

</hfoption>
</hfoptions>

## YolosConfig

[[autodoc]] YolosConfig

## YolosImageProcessor

[[autodoc]] YolosImageProcessor
    - preprocess
    - pad
    - post_process_object_detection

## YolosImageProcessorFast

[[autodoc]] YolosImageProcessorFast
    - preprocess
    - pad
    - post_process_object_detection

## YolosFeatureExtractor

[[autodoc]] YolosFeatureExtractor
    - __call__
    - pad
    - post_process_object_detection

## YolosModel

[[autodoc]] YolosModel
    - forward

## YolosForObjectDetection

[[autodoc]] YolosForObjectDetection
    - forward

