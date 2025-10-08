<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-04-05 and added to Hugging Face Transformers on 2023-04-19 and contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).*

# SAM

[Segment Anything Model](https://huggingface.co/papers/2304.02643v1) introduces the Segment Anything (SA) project, encompassing a novel task, model, and dataset for image segmentation. Leveraging an efficient model in a data collection loop, the project created the largest segmentation dataset to date, featuring over 1 billion masks across 11 million licensed and privacy-respecting images. The model is designed to be promptable, enabling zero-shot transfer to new image distributions and tasks. Evaluations demonstrate its impressive zero-shot performance, often matching or surpassing fully supervised results. The Segment Anything Model (SAM) and the corresponding dataset (SA-1B) are available for fostering research in foundation models for computer vision. The model predicts binary masks indicating the presence of objects in images and performs better with input 2D points and/or bounding boxes. Multiple points can prompt a single mask, and while fine-tuning is unsupported, textual input is planned.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline

generator = pipeline(task="mask-generation", model="facebook/sam-vit-huge", points_per_batch=256, dtype="auto")
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
outputs = generator(image_url, points_per_batch = 256)

raw_image = Image.open(requests.get(image_url, stream=True).raw)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

plt.imshow(np.array(raw_image))
ax = plt.gca()
for mask in outputs["masks"]:
    show_mask(mask, ax=ax, random_color=True)
plt.axis("off")
plt.show()
```

</hfoption>
</hfoptions>

## SamConfig

[[autodoc]] SamConfig

## SamVisionConfig

[[autodoc]] SamVisionConfig

## SamMaskDecoderConfig

[[autodoc]] SamMaskDecoderConfig

## SamPromptEncoderConfig

[[autodoc]] SamPromptEncoderConfig

## SamProcessor

[[autodoc]] SamProcessor

## SamImageProcessor

[[autodoc]] SamImageProcessor

## SamImageProcessorFast

[[autodoc]] SamImageProcessorFast

## SamModel

[[autodoc]] SamModel
    - forward

## SamVisionModel

[[autodoc]] SamVisionModel
    - forward

