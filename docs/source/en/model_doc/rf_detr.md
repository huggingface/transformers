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
*This model was released on 2024-04-05 and added to Hugging Face Transformers on 2026-01-23.*

<div style="float: right;">
 <div class="flex flex-wrap space-x-1">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
 </div>
</div>

# RF-DETR

[RF-DETR](https://huggingface.co/papers/2407.17140) proposes a Receptive Field Detection Transformer (DETR) architecture
designed to compete with and surpass the dominant YOLO series for real-time object detection. It achieves a new
state-of-the-art balance between speed (latency) and accuracy (mAP) by combining recent transformer advances with
efficient design choices.

The RF-DETR architecture is characterized by its simple and efficient structure: a DINOv2 Backbone, a Projector, and a
shallow DETR Decoder.
It enhances the DETR architecture for efficiency and speed using the following core modifications:

1. **DINOv2 Backbone**: Uses a powerful DINOv2 backbone for robust feature extraction.
2. **Group DETR Training**: Utilizes Group-Wise One-to-Many Assignment during training to accelerate convergence.
3. **Richer Input**: Aggregates multi-level features from the backbone and uses a C2f Projector (similarly to YOLOv8) to
   pass multi-scale features.
4. **Faster Decoder**: Employs a shallow 3-layer DETR decoder with deformable cross-attention for lower latency.
5. **Optimized Queries**: Uses a mixed-query scheme combining learnable content queries and generated spatial queries.

You can find all the available RF-DETR checkpoints under the [stevenbucaille](https://huggingface.co/stevenbucaille)
organization.
The original code can be found [here](https://github.com/roboflow/rf-detr).

> [!TIP]
> This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
>
> Click on the RF-DETR models in the right sidebar for more examples of how to apply RF-DETR to different object
> detection tasks.


The example below demonstrates how to perform object detection with the [`Pipeline`] and the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline
import torch

pipeline = pipeline(
    "object-detection",
    model="stevenbucaille/rfdetr_small_60e_coco",
    dtype=torch.float16,
    device_map=0
)

pipeline("http://images.cocodataset.org/val2017/000000039769.jpg")
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image
import requests
import torch

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("stevenbucaille/rfdetr_small")
model = AutoModelForObjectDetection.from_pretrained("stevenbucaille/rfdetr_small")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{model.config.id2label[label]}: {score:.2f} {box}")
```

</hfoption>
</hfoptions>

## Resources


- Scripts for finetuning [`RfDetrForObjectDetection`] with [`Trainer`]
  or [Accelerate](https://huggingface.co/docs/accelerate/index) can be
  found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- See also: [Object detection task guide](../tasks/object_detection).

## RfDetrConfig

[[autodoc]] RfDetrConfig

## RfDetrDinov2Config

[[autodoc]] RfDetrDinov2Config

## RfDetrModel

[[autodoc]] RfDetrModel
    - forward

## RfDetrForObjectDetection

[[autodoc]] RfDetrForObjectDetection
    - forward

## RfDetrForInstanceSegmentation

[[autodoc]] RfDetrForInstanceSegmentation
    - forward

## RfDetrDinov2Backbone

[[autodoc]] RfDetrDinov2Backbone
    - forward
