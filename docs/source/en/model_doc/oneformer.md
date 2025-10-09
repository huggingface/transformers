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
*This model was released on 2022-11-10 and added to Hugging Face Transformers on 2023-01-19 and contributed by [praeclarumjj3](https://huggingface.co/praeclarumjj3).*

# OneFormer

[OneFormer: One Transformer to Rule Universal Image Segmentation](https://huggingface.co/papers/2211.06220) is a universal image segmentation framework capable of performing semantic, instance, and panoptic segmentation tasks after being trained on a single panoptic dataset. It employs a task token to condition the model on the specific task during inference, enabling task-dynamic behavior. The model uses a task-conditioned joint training strategy and a query-text contrastive loss to enhance inter-task and inter-class distinctions. OneFormer outperforms specialized Mask2Former models across all three segmentation tasks on datasets like ADE20k, CityScapes, and COCO, even with fewer resources. Enhanced performance is also observed with new ConvNeXt and DiNAT backbones.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-segmentation", model="shi-labs/oneformer_ade20k_swin_tiny", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import OneFormerProcessor, AutoModelForUniversalSegmentation
from PIL import Image

processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", dtype="auto")

url = ("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, ["panoptic"], return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

predicted_panoptic_map = processor.post_process_panoptic_segmentation(
    outputs, target_sizes=[(image.height, image.width)]
)[0]["segmentation"]

plt.figure(figsize=(8, 6))
plt.imshow(predicted_panoptic_map, cmap='tab20')
plt.axis('off')
plt.show()
```

</hfoption>
</hfoptions>

## OneFormer specific outputs

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerModelOutput

[[autodoc]] models.oneformer.modeling_oneformer.OneFormerForUniversalSegmentationOutput

## OneFormerConfig

[[autodoc]] OneFormerConfig

## OneFormerImageProcessor

[[autodoc]] OneFormerImageProcessor
    - preprocess
    - encode_inputs
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## OneFormerImageProcessorFast

[[autodoc]] OneFormerImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation
    - post_process_instance_segmentation
    - post_process_panoptic_segmentation

## OneFormerProcessor

[[autodoc]] OneFormerProcessor

## OneFormerModel

[[autodoc]] OneFormerModel
    - forward

## OneFormerForUniversalSegmentation

[[autodoc]] OneFormerForUniversalSegmentation
    - forward
    

