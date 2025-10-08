<!--Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

*This model was released on 2025-03-24 and added to Hugging Face Transformers on 2025-06-27 and contributed by [yaswanthgali](https://huggingface.co/yaswanthgali).*

# EoMT

[Encoder-only Mask Transformer](https://huggingface.co/papers/2503.19108) (EoMT) repurposes the plain Vision Transformer (ViT) architecture for image segmentation without task-specific components. By leveraging large-scale models and extensive pre-training, EoMT achieves segmentation accuracy comparable to state-of-the-art methods while being significantly faster due to its simpler architecture. This approach suggests that scaling the ViT itself is more effective than adding architectural complexity.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="tue-mps/coco_panoptic_eomt_large_640", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor


processor = AutoImageProcessor.from_pretrained("tue-mps/coco_panoptic_eomt_large_640")
model = AutoModelForUniversalSegmentation.from_pretrained("tue-mps/coco_panoptic_eomt_large_640", dtype="auto")

image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", stream=True).raw)
inputs = processor(images=image, return_tensors="pt",)

with torch.inference_mode():
    outputs = model(**inputs)

target_sizes = [(image.height, image.width)]
outputs = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

plt.imshow(outputs[0]["segmentation"])
plt.axis("off")
plt.show()
```

</hfoption>
</hfoptions>

## EomtImageProcessor

[[autodoc]] EomtImageProcessor
    - preprocess
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## EomtImageProcessorFast

[[autodoc]] EomtImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## EomtConfig

[[autodoc]] EomtConfig

## EomtForUniversalSegmentation

[[autodoc]] EomtForUniversalSegmentation
    - forward

