<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ViTMatte

## Overview

The ViTMatte model was proposed in [Boosting Image Matting with Pretrained Plain Vision Transformers](https://arxiv.org/abs/2305.15272) by Jingfeng Yao, Xinggang Wang, Shusheng Yang, Baoyuan Wang.
ViTMatte leverages plain [Vision Transformers](vit) for the task of image matting, which is the process of accurately estimating the foreground object in images and videos.

The abstract from the paper is the following:

*Recently, plain vision Transformers (ViTs) have shown impressive performance on various computer vision tasks, thanks to their strong modeling capacity and large-scale pretraining. However, they have not yet conquered the problem of image matting. We hypothesize that image matting could also be boosted by ViTs and present a new efficient and robust ViT-based matting system, named ViTMatte. Our method utilizes (i) a hybrid attention mechanism combined with a convolution neck to help ViTs achieve an excellent performance-computation trade-off in matting tasks. (ii) Additionally, we introduce the detail capture module, which just consists of simple lightweight convolutions to complement the detailed information required by matting. To the best of our knowledge, ViTMatte is the first work to unleash the potential of ViT on image matting with concise adaptation. It inherits many superior properties from ViT to matting, including various pretraining strategies, concise architecture design, and flexible inference strategies. We evaluate ViTMatte on Composition-1k and Distinctions-646, the most commonly used benchmark for image matting, our method achieves state-of-the-art performance and outperforms prior matting works by a large margin.*

Tips:

- The model expects both the image and trimap (concatenated) as input. One can use [`ViTMatteImageProcessor`] for this purpose.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/hustvl/ViTMatte).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vitmatte_architecture.png"
alt="drawing" width="600"/>

<small> ViTMatte high-level overview. Taken from the <a href="https://arxiv.org/abs/2305.15272">original paper.</a> </small>

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViTMatte.

- A demo notebook regarding inference with [`VitMatteForImageMatting`], including background replacement, can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViTMatte).


## VitMatteConfig

[[autodoc]] VitMatteConfig

## VitMatteImageProcessor

[[autodoc]] VitMatteImageProcessor
    - preprocess

## VitMatteForImageMatting

[[autodoc]] VitMatteForImageMatting
    - forward