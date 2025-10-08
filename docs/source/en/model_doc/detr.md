<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-05-26 and added to Hugging Face Transformers on 2021-06-09 and contributed by [nielsr](https://huggingface.co/nielsr).*

# DETR

[DETR](https://huggingface.co/papers/2005.12872) presents a novel method for object detection by framing it as a direct set prediction problem. This approach eliminates the need for hand-designed components such as non-maximum suppression and anchor generation. DETR uses a set-based global loss and a transformer encoder-decoder architecture to output predictions in parallel. It achieves accuracy and runtime performance comparable to Faster R-CNN on the COCO dataset and can be extended to panoptic segmentation with superior results.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="facebook/detr-resnet-50", dtype="auto")
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
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50", dtype="auto")

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

## DetrConfig

[[autodoc]] DetrConfig

## DetrImageProcessor

[[autodoc]] DetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DetrImageProcessorFast

[[autodoc]] DetrImageProcessorFast
    - preprocess
    - post_process_object_detection
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## DETR specific outputs

[[autodoc]] models.detr.modeling_detr.DetrModelOutput

[[autodoc]] models.detr.modeling_detr.DetrObjectDetectionOutput

[[autodoc]] models.detr.modeling_detr.DetrSegmentationOutput

## DetrModel

[[autodoc]] DetrModel
    - forward

## DetrForObjectDetection

[[autodoc]] DetrForObjectDetection
    - forward

## DetrForSegmentation

[[autodoc]] DetrForSegmentation
    - forward

