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
*This model was released on 2020-10-08 and added to Hugging Face Transformers on 2022-09-14 and contributed by [nielsr](https://huggingface.co/nielsr).*

# Deformable DETR

[Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://huggingface.co/papers/2010.04159) addresses the slow convergence and limited feature spatial resolution issues of DETR by introducing a deformable attention module. This module focuses on a small set of key sampling points around a reference, enhancing performance, particularly for small objects, and reducing training time by a factor of ten. Experiments on the COCO benchmark confirm the effectiveness of this approach.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="SenseTime/deformable-detr", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
model = AutoModelForObjectDetection.from_pretrained("SenseTime/deformable-detr", dtype="auto")

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

## DeformableDetrImageProcessor

[[autodoc]] DeformableDetrImageProcessor
    - preprocess
    - post_process_object_detection

## DeformableDetrImageProcessorFast

[[autodoc]] DeformableDetrImageProcessorFast
    - preprocess
    - post_process_object_detection

## DeformableDetrFeatureExtractor

[[autodoc]] DeformableDetrFeatureExtractor
    - __call__
    - post_process_object_detection

## DeformableDetrConfig

[[autodoc]] DeformableDetrConfig

## DeformableDetrModel

[[autodoc]] DeformableDetrModel
    - forward

## DeformableDetrForObjectDetection

[[autodoc]] DeformableDetrForObjectDetection
    - forward

