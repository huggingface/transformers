<!--Copyright 2026 Mobile Perception Systems Lab at TU/e and The Hugging Face team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

*This model was released on 2025-09-09 and added to Hugging Face Transformers on 2026-01-27.*

# EoMT-DINOv3

<div class="flex flex-wrap space-x-1">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
  <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The **EoMT-DINOv3** family extends the [Encoder-only Mask Transformer](eomt) architecture with
Vision Transformers that are pre-trained using [DINOv3](dinov3). The update delivers stronger segmentation quality across ADE20K and COCO
benchmarks while preserving the encoder-only design that made EoMT attractive for real-time applications.

Compared to the DINOv2-based models, the DINOv3 variants leverage rotary position embeddings, optional gated MLP blocks
and the latest pre-training recipes from Meta AI. These changes yield measurable performance gains across semantic,
instance and panoptic segmentation tasks, as highlighted in the [DINOv3 model zoo](https://github.com/tue-mps/eomt/blob/master/model_zoo/dinov3.md).

The original EoMT architecture was introduced in the CVPR 2025 Highlight paper *[Your ViT is Secretly an Image
Segmentation Model](https://huggingface.co/papers/2503.19108)* by Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans,
Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman and Daan de Geus. The DINOv3 upgrade keeps the same
lightweight segmentation head and query-based inference strategy while swapping the encoder for DINOv3 ViT checkpoints.

Tips:

* The configuration exposes DINOv3-specific knobs such as `rope_theta` and `use_gated_mlp`. Large DINOv3 backbones
  such as `dinov3-vitg14` expect `use_gated_mlp=True`.
* DINOv3 models can operate on a broader range of resolutions thanks to rotary position embeddings. The image processor
  still defaults to square crops but custom sizes can be supplied through `AutoImageProcessor`.
* The pre-trained checkpoints hosted by the TU/e Mobile Perception Systems Lab provide delta weights that should be
  combined with the upstream DINOv3 backbones. The conversion utilities in the
  [official repository](https://github.com/tue-mps/eomt) describe this workflow in detail.

This model was contributed by [NielsRogge](https://huggingface.co/NielsRogge).
The original code can be found [here](https://github.com/tue-mps/eomt).

## Usage examples

Below is a minimal example showing how to run panoptic segmentation with a DINOv3-backed EoMT model. The same
image processor can be reused for semantic or instance segmentation simply by swapping the checkpoint.

```python
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation


model_id = "tue-mps/eomt-dinov3-coco-panoptic-base-640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForUniversalSegmentation.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(images=image, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)

segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

plt.imshow(segmentation["segmentation"])
plt.axis("off")
plt.show()
```

## EomtDinov3Config

[[autodoc]] EomtDinov3Config

## EomtDinov3PreTrainedModel

[[autodoc]] EomtDinov3PreTrainedModel
    - forward

## EomtDinov3ForUniversalSegmentation

[[autodoc]] EomtDinov3ForUniversalSegmentation
