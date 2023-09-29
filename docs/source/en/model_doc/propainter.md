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

# ProPainter

## Overview

ProPainter model was proposed in [ProPainter: Improving Propagation and Transformer for Video Inpainting](https://arxiv.org/abs/2309.03897) by Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan and Chen Change Loy.

The abstract from the paper is the following:

Flow-based propagation and spatiotemporal Transformer are two mainstream mechanisms in video inpainting (VI). Despite the effectiveness of these components, they still suffer from some limitations that affect their performance. Previous propagation-based approaches are performed separately either in the image or feature domain. Global image propagation isolated from learning may cause spatial misalignment due to inaccurate optical flow. Moreover, memory or computational constraints limit the temporal range of feature propagation and video Transformer, preventing exploration of correspondence information from distant frames. To address these issues, we propose an improved framework, called ProPainter, which involves enhanced ProPagation and an efficient Transformer. Specifically, we introduce dual-domain propagation that combines the advantages of image and feature warping, exploiting global correspondences reliably. We also propose a mask-guided sparse video Transformer, which achieves high efficiency by discarding unnecessary and redundant tokens. With these components, ProPainter outperforms prior arts by a large margin of 1.46 dB in PSNR while maintaining appealing efficiency.


<img src="https://shangchenzhou.com/projects/ProPainter/assets/images/ProPainter_pipeline.png"
alt="drawing" width="600"/>

Checkout the model [here](https://huggingface.co/models?search=llava-hf)

Tips:

- Weights for the ProPainter can be obtained from [here](https://huggingface.co/shauray/ProPainter-hf/)

Table shows the estimated GPU memory requirements for different sub-video lengths with fp32/fp16 precision: 

| Resolution | 50 frames | 80 frames |
| :---       | :----:    | :----:    |
| 1280 x 720 | 28G / 19G | OOM / 25G |
| 720 x 480  | 11G / 7G  | 13G / 8G  |
| 640 x 480  | 10G / 6G  | 12G / 7G  |
| 320 x 240  | 3G  / 2G  | 4G  / 3G  |

This model was contributed by [Shauray Singh](https://huggingface.co/shauray) The original code of the authors can be found [here](https://github.com/sczhou/ProPainter/).

## ProPainterConfig

[[autodoc]] ProPainterConfig

## ProPainterImageProcessor

[[autodoc]] ProPainterImageProcessor
    - preprocess

## ProPainterModel

[[autodoc]] ProPainterModel
    - forward

## ProPainterForImageInPainting

[[autodoc]] ProPainterForImageInPainting
    - forward

## ProPainterForImageOutPainting

[[autodoc]] ProPainterForImageOutPainting
    - forward

