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
*This model was released on 2025-11-13 and added to Hugging Face Transformers on 2026-03-02.*

<div style="float: right;">
 <div class="flex flex-wrap space-x-1">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
 </div>
</div>

# RF-DETR

[RF-DETR](https://huggingface.co/papers/2511.09554) is an end-to-end detector built on top of a DINOv2-style vision backbone and a lightweight DETR decoder. In this Transformers implementation, RF-DETR uses a windowed DINOv2-with-registers backbone and a Group-DETR-style decoder stack inherited from LW-DETR components for efficient object detection and instance segmentation.

The original project is available at [roboflow/rf-detr](https://github.com/roboflow/rf-detr).

> [!TIP]
> This model was contributed by [nielsr](https://huggingface.co/nielsr).
>
> Click on RF-DETR models in the right sidebar for more examples.

The example below demonstrates how to perform object detection with the [`Pipeline`] and [`AutoModelForObjectDetection`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "object-detection",
    model="nielsr/rf-detr-small",
    dtype=torch.float16,
    device_map=0,
)

outputs = pipe("http://images.cocodataset.org/val2017/000000039769.jpg")
print(outputs[:3])
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

image_processor = AutoImageProcessor.from_pretrained("nielsr/rf-detr-small")
model = AutoModelForObjectDetection.from_pretrained("nielsr/rf-detr-small")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=0.3,
)[0]

for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
    score = score.item()
    label = model.config.id2label[label_id.item()]
    box = [round(i, 2) for i in box.tolist()]
    print(f"{label}: {score:.2f} {box}")
```

</hfoption>
</hfoptions>

### Instance segmentation

```python
from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation
from PIL import Image
import requests
import torch

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("nielsr/rf-detr-seg-small")
model = RfDetrForInstanceSegmentation.from_pretrained("nielsr/rf-detr-seg-small")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = image_processor.post_process_instance_segmentation(
    outputs,
    target_sizes=torch.tensor([image.size[::-1]]),
    threshold=0.3,
)[0]

print(results.keys())
print(results["scores"][:3])
print(results["labels"][:3])
print(results["masks"].shape)
```

## Notes

- RF-DETR predicts detections from `num_queries * group_detr` query slots during training/inference (`300 * 13 = 3900` by default).
- The backbone is configured via [`~transformers.RfDetrWindowedDinov2Config`] and can be instantiated as a standalone backbone with [`~transformers.RfDetrWindowedDinov2Backbone`].
- This implementation exposes both object detection ([`~transformers.RfDetrForObjectDetection`]) and instance segmentation ([`~transformers.RfDetrForInstanceSegmentation`]).
- The instance segmentation head currently supports inference. Training loss for [`~transformers.RfDetrForInstanceSegmentation`] is not implemented yet.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RF-DETR.

<PipelineTag pipeline="object-detection"/>
<PipelineTag pipeline="image-segmentation"/>

- Scripts for fine-tuning DETR-like object detection models with [`Trainer`] or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- See also: [Object detection task guide](../tasks/object_detection).

## RfDetrConfig

[[autodoc]] RfDetrConfig

## RfDetrWindowedDinov2Config

[[autodoc]] RfDetrWindowedDinov2Config

## RF-DETR specific outputs

[[autodoc]] models.rf_detr.modeling_rf_detr.RfDetrModelOutput

[[autodoc]] models.rf_detr.modeling_rf_detr.RfDetrObjectDetectionOutput

[[autodoc]] models.rf_detr.modeling_rf_detr.RfDetrInstanceSegmentationOutput

## RfDetrModel

[[autodoc]] RfDetrModel
    - forward

## RfDetrForObjectDetection

[[autodoc]] RfDetrForObjectDetection
    - forward

## RfDetrForInstanceSegmentation

[[autodoc]] RfDetrForInstanceSegmentation
    - forward

## RfDetrWindowedDinov2Backbone

[[autodoc]] RfDetrWindowedDinov2Backbone
    - forward

## RfDetrImageProcessor

[[autodoc]] RfDetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_instance_segmentation

## RfDetrImageProcessorFast

[[autodoc]] RfDetrImageProcessorFast
    - preprocess
    - post_process_object_detection
    - post_process_instance_segmentation
