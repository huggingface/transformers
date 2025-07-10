<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# HGNet-V2

## Overview

A HGNet-V2 (High Performance GPU Net) image classification model.
HGNet arhtictecture was proposed in [HGNET: A Hierarchical Feature Guided Network for Occupancy Flow Field Prediction](https://huggingface.co/papers/2407.01097) by
Zhan Chen, Chen Tang, Lu Xiong

The abstract from the HGNET paper is the following:

*Predicting the motion of multiple traffic participants has always been one of the most challenging tasks in autonomous driving. The recently proposed occupancy flow field prediction method has shown to be a more effective and scalable representation compared to general trajectory prediction methods. However, in complex multi-agent traffic scenarios, it remains difficult to model the interactions among various factors and the dependencies among prediction outputs at different time steps. In view of this, we propose a transformer-based hierarchical feature guided network (HGNET), which can efficiently extract features of agents and map information from visual and vectorized inputs, modeling multimodal interaction relationships. Second, we design the Feature-Guided Attention (FGAT) module to leverage the potential guiding effects between different prediction targets, thereby improving prediction accuracy. Additionally, to enhance the temporal consistency and causal relationships of the predictions, we propose a Time Series Memory framework to learn the conditional distribution models of the prediction outputs at future time steps from multivariate time series. The results demonstrate that our model exhibits competitive performance, which ranks 3rd in the 2024 Waymo Occupancy and Flow Prediction Challenge.*

This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber). 
The original code can be found [here](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py).

## HGNetV2Config

[[autodoc]] HGNetV2Config


## HGNetV2Backbone

[[autodoc]] HGNetV2Backbone
    - forward


## HGNetV2ForImageClassification

[[autodoc]] HGNetV2ForImageClassification
    - forward