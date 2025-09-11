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
*This model was released on 2025-03-24 and added to Hugging Face Transformers on 2025-06-27.*

# EoMT

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[The Encoder-only Mask Transformer]((https://www.tue-mps.org/eomt)) (EoMT) model was introduced in the CVPR 2025 Highlight Paper *[Your ViT is Secretly an Image Segmentation Model](https://huggingface.co/papers/2503.19108)* by Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, and Daan de Geus.
EoMT reveals Vision Transformers can perform image segmentation efficiently without task-specific components.

The abstract from the paper is the following:

*Vision Transformers (ViTs) have shown remarkable performance and scalability across various computer vision tasks. To apply single-scale ViTs to image segmentation, existing methods adopt a convolutional adapter to generate multi-scale features, a pixel decoder to fuse these features, and a Transformer decoder that uses the fused features to make predictions. In this paper, we show that the inductive biases introduced by these task-specific components can instead be learned by the ViT itself, given sufficiently large models and extensive pre-training. Based on these findings, we introduce the Encoder-only Mask Transformer (EoMT), which repurposes the plain ViT architecture to conduct image segmentation. With large-scale models and pre-training, EoMT obtains a segmentation accuracy similar to state-of-the-art models that use task-specific components. At the same time, EoMT is significantly faster than these methods due to its architectural simplicity, e.g., up to 4x faster with ViT-L. Across a range of model sizes, EoMT demonstrates an optimal balance between segmentation accuracy and prediction speed, suggesting that compute resources are better spent on scaling the ViT itself rather than adding architectural complexity.*

This model was contributed by [Yaswanth Gali](https://huggingface.co/yaswanthgali).
The original code can be found [here](https://github.com/tue-mps/eomt).

## Architecture Info

The `EoMT` model uses a DINOv2-pretrained Vision Transformer with **register tokens** as its backbone. EoMT simplifies the segmentation pipeline by relying solely on the encoder, eliminating the need for task-specific decoders commonly used in prior approaches.

Architecturally, EoMT introduces a small set of **learned queries** and a lightweight **mask prediction module**. These queries are injected into the final encoder blocks, enabling **joint attention** between image patches and object queries. During training, **masked attention** is applied to constrain each query to focus on its corresponding region—effectively mimicking cross-attention. This constraint is gradually phased out via a **mask annealing strategy**, allowing for **efficient, decoder-free inference** without compromising segmentation performance.

<div style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/eomt_architecture.png"
       alt="drawing" width="500"/>
</div>


The model supports semantic, instance, and panoptic segmentation using a unified architecture and task-specific post-processing.

## Usage Examples

Use the Hugging Face implementation of EoMT for inference with pre-trained models.

### Semantic Segmentation

The EoMT model performs semantic segmentation using sliding-window inference. The input image is resized such that the shorter side matches the target input size, then it is split into overlapping crops. Each crop is then passed through the model. After inference, the predicted logits from each crop are stitched back together and rescaled to the original image size to get the final segmentation mask.

> **Note:**  
> If you want to use a custom target size for **semantic segmentation**, specify it in the following format:  
> `{"shortest_edge": 512}`  
> Notice that `longest_edge` is not provided here — this is intentional. For semantic segmentation, images are typically **scaled so that the shortest edge is greater than or equal to the target size** hence longest_edge is not necessary.

```python
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from transformers import EomtForUniversalSegmentation, AutoImageProcessor


model_id = "tue-mps/ade20k_semantic_eomt_large_512"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
target_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

# Visualize the segmentation mask
plt.imshow(preds[0])
plt.axis("off")
plt.title("Semantic Segmentation")
plt.show()
```

### Instance Segmentation

The EoMT model performs instance segmentation using padded inference. The input image is resized so that the longer side matches the target input size, and the shorter side is zero-padded to form a square. The resulting mask and class logits are combined through post-processing (adapted from Mask2Former) to produce a unified instance segmentation map, along with segment metadata like segment id, class labels and confidence scores.

> **Note:**  
> To use a custom target size, specify the size as a dictionary in the following format:  
> `{"shortest_edge": 512, "longest_edge": 512}`  
> For both instance and panoptic segmentation, input images will be **scaled and padded** to this target size.

```python
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from transformers import EomtForUniversalSegmentation, AutoImageProcessor


model_id = "tue-mps/coco_instance_eomt_large_640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
target_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_instance_segmentation(
    outputs,
    target_sizes=target_sizes,
)

# Visualize the segmentation mask
plt.imshow(preds[0]["segmentation"])
plt.axis("off")
plt.title("Instance Segmentation")
plt.show()
```

### Panoptic Segmentation

The EoMT model performs panoptic segmentation using the same padded inference strategy as in instance segmentation. After padding and normalization, the model predicts both thing (instances) and stuff (amorphous regions) classes. The resulting mask and class logits are combined through post-processing (adapted from Mask2Former) to produce a unified panoptic segmentation map, along with segment metadata like segment id, class labels and confidence scores.

```python
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from transformers import EomtForUniversalSegmentation, AutoImageProcessor


model_id = "tue-mps/coco_panoptic_eomt_large_640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
target_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

# Visualize the panoptic segmentation mask
plt.imshow(preds[0]["segmentation"])
plt.axis("off")
plt.title("Panoptic Segmentation")
plt.show()
```

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