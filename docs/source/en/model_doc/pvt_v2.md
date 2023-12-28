<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Pyramid Vision Transformer V2 (PVT v2)

## Overview

The PVT v2 model was proposed in
[PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797) by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. As an improved variant of PVT v1, it eschews positional encoding, relying instead on positional  information encoded through zero-padding and overlapping patch embeddings. The lack of reliance on positional encoding simplifies the architecture and avoids the need for interpolating positional encodings. The same encoder structure is used in [Segformer (https://arxiv.org/abs/2105.15203) for semantic segmentation.

Abstract from the paper:

*Transformer recently has presented encouraging progress in computer vision. In this work, we present new baselines by improving the original Pyramid Vision Transformer (PVT v1) by adding three designs, including (1) linear complexity attention layer, (2) overlapping patch embedding, and (3) convolutional feed-forward network. With these modifications, PVT v2 reduces the computational complexity of PVT v1 to linear and achieves significant improvements on fundamental vision tasks such as classification, detection, and segmentation. Notably, the proposed PVT v2 achieves comparable or better performances than recent works such as Swin Transformer. We hope this work will facilitate state-of-the-art Transformer researches in computer vision. Code is available at https://github.com/whai362/PVT.*

This model was contributed by [FoamoftheSea](https://huggingface.co/FoamoftheSea). The original code can be found [here](https://github.com/whai362/PVT).

## Usage tips

- [PVTv2](https://arxiv.org/abs/2106.13797) is a hierarchical transformer model which has demonstrated powerful performance in image classification and multiple other tasks, used as a backbone for semantic segmentation in [Segformer](https://arxiv.org/abs/2105.15203), monocular depth estimation in [GLPN](https://arxiv.org/abs/2201.07436), and panoptic segmentation in [Panoptic Segformer](https://arxiv.org/abs/2109.03814).
- Hierarchical transformers like PVTv2 achieve superior data and parameter efficiency on image data compared with pure transformer architectures by incorporating convolution operations into the encoder layers. This creates a best-of-both-worlds architecture that infuses the useful inductive biases of CNNs like translation equivariance and locality into the network while still enjoying the benefits of dynamic data response and global relationship modeling provided by the self-attention mechanism of [transformers](https://arxiv.org/abs/1706.03762).
- PVTv2 uses overlapping patch embeddings to create multi-scale feature maps, which are infused with location information using zero-padding and depth-wise convolutions.
- To reduce the complexity in the attention layers, PVTv2 has the option of using strided 2D convolution or fixed-size average pooling on the intermediary feature maps, the latter of which provides impressive performance with a linear complexity with respect to image size. To use the average pooling for linear complexity, set `linear_attention=True` in the `PvtV2Config`.
- [`PvtV2Model`] is the hierarchical Transformer encoder (which is also often referred to as Mix Transformer or MiT in the literature). [`PvtV2ForImageClassification`] adds a simple classifier head on top to perform Image Classification.
- ImageNet pretrained weights for all model sizes can be found on the [hub](https://huggingface.co/models?other=pvt_v2).
- The best way to get started with the PVTv2 is to load the pretrained checkpoint with the size of your choosing using `AutoModelForImageClassification`:
```python
import requests
import torch

from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

model = AutoModelForImageClassification.from_pretrained("FoamoftheSea/pvt_v2_b0")
image_processor = AutoImageProcessor.from_pretrained("FoamoftheSea/pvt_v2_b0")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
processed = image_processor(image)
outputs = model(torch.tensor(processed["pixel_values"]))
```

[PVTv2](https://github.com/whai362/PVT/tree/v2) performance on ImageNet-1K by model size (B0-B5):

| Method           | Size | Acc@1 | #Params (M) |
|------------------|:----:|:-----:|:-----------:|
| PVT-V2-B0        |  224 |  70.5 |     3.7     |
| PVT-V2-B1        |  224 |  78.7 |     14.0    |
| PVT-V2-B2-Linear |  224 |  82.1 |     22.6    |
| PVT-V2-B2        |  224 |  82.0 |     25.4    |
| PVT-V2-B3        |  224 |  83.1 |     45.2    |
| PVT-V2-B4        |  224 |  83.6 |     62.6    |
| PVT-V2-B5        |  224 |  83.8 |     82.0    |


## PvtV2Config

[[autodoc]] PvtV2Config

## PvtForImageClassification

[[autodoc]] PvtV2ForImageClassification
    - forward

## PvtModel

[[autodoc]] PvtV2Model
    - forward
