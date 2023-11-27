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
[PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)
by Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao.
As an improved variant of PVT v1, it eschews positional encoding, relying instead on positional 
information encoded through zero-padding and overlapping patch embeddings. The lack of reliance on
positional encoding simplifies the architecture and avoids the need for interpolating positional
encodings. The same encoder structure is used in [Segformer](https://arxiv.org/abs/2105.15203) for
semantic segmentation.

Abstract from the paper:

*Transformer recently has presented encouraging
progress in computer vision. In this work, we present
new baselines by improving the original Pyramid Vision
Transformer (PVT v1) by adding three designs, including
(1) linear complexity attention layer, (2) overlapping
patch embedding, and (3) convolutional feed-forward
network. With these modifications, PVT v2 reduces the
computational complexity of PVT v1 to linear and achieves
significant improvements on fundamental vision tasks such
as classification, detection, and segmentation. Notably, the
proposed PVT v2 achieves comparable or better performances than recent works such as Swin Transformer. We
hope this work will facilitate state-of-the-art Transformer
researches in computer vision. Code is available at
https://github.com/whai362/PVT.*

This model was contributed by [FoamoftheSea](https://huggingface.co/FoamoftheSea). The original code can be found [here](https://github.com/whai362/PVT).


- PVTv2 on ImageNet-1K

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
