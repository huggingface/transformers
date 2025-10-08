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
*This model was released on 2021-08-13 and added to Hugging Face Transformers on 2022-09-22 and contributed by [DepuMeng](https://huggingface.co/DepuMeng).*

# Conditional DETR

[Conditional DETR](https://huggingface.co/papers/2108.06152) addresses slow training convergence in DETR by introducing a conditional cross-attention mechanism. This mechanism allows the decoder to learn a conditional spatial query, enabling each cross-attention head to focus on distinct regions such as object extremities or internal regions. This approach reduces reliance on high-quality content embeddings, simplifying training and achieving up to 10× faster convergence for stronger backbones.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="object-detection", model="microsoft/conditional-detr-resnet-50", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50", dtype="auto")

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

## ConditionalDetrConfig

[[autodoc]] ConditionalDetrConfig

## ConditionalDetrImageProcessor

[[autodoc]] ConditionalDetrImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_instance_segmentation
    - post_process_semantic_segmentation
    - post_process_panoptic_segmentation

## ConditionalDetrImageProcessorFast

[[autodoc]] ConditionalDetrImageProcessorFast
    - preprocess
    - post_process_object_detection
    - post_process_instance_segmentation
    - post_process_semantic_segmentation
    - post_process_panoptic_segmentation

## ConditionalDetrFeatureExtractor

[[autodoc]] ConditionalDetrFeatureExtractor
    - __call__
    - post_process_object_detection
    - post_process_instance_segmentation
    - post_process_semantic_segmentation
    - post_process_panoptic_segmentation

## ConditionalDetrModel

[[autodoc]] ConditionalDetrModel
    - forward

## ConditionalDetrForObjectDetection

[[autodoc]] ConditionalDetrForObjectDetection
    - forward

## ConditionalDetrForSegmentation

[[autodoc]] ConditionalDetrForSegmentation
    - forward

