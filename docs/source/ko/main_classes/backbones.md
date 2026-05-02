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

# 백본 (Backbone) [[backbone]] 

백본(backbone)은 객체 탐지나 이미지 분류와 같은 상위 수준의 컴퓨터 비전 작업을 위해 특성(feature) 추출에 사용되는 모델을 말합니다. Transformers 라이브러리는 사전 학습된 모델 가중치로부터 Transformers 백본을 초기화할 수 있는 [`AutoBackbone`] 클래스를 제공하며, 다음 두 가지 유틸리티 클래스도 포함합니다:

* [`~utils.BackboneMixin`]은 Transformers 또는 [timm](https://hf.co/docs/timm/index)에서 백본을 초기화할 수 있게 해주며, 출력 특성과 인덱스를 반환하는 기능을 포함합니다.
* [`~utils.BackboneConfigMixin`]은 백본 설정의 출력 특성과 인덱스를 지정합니다.

[timm](https://hf.co/docs/timm/index) 모델은 [`TimmBackbone`]과 [`TimmBackboneConfig`] 클래스를 사용하여 불러옵니다.

백본은 다음 모델들에서 지원됩니다:

* [BEiT](../model_doc/beit)
* [BiT](../model_doc/bit)
* [ConvNext](../model_doc/convnext)
* [ConvNextV2](../model_doc/convnextv2)
* [DiNAT](../model_doc/dinat)
* [DINOV2](../model_doc/dinov2)
* [FocalNet](../model_doc/focalnet)
* [MaskFormer](../model_doc/maskformer)
* [NAT](../model_doc/nat)
* [ResNet](../model_doc/resnet)
* [Swin Transformer](../model_doc/swin)
* [Swin Transformer v2](../model_doc/swinv2)
* [ViTDet](../model_doc/vitdet)

## AutoBackbone [[transformers.AutoBackbone]]

[[autodoc]] AutoBackbone

## BackboneMixin [[transformers.utils.BackboneMixin]]

[[autodoc]] utils.BackboneMixin

## BackboneConfigMixin [[transformers.utils.BackboneConfigMixin]]

[[autodoc]] utils.BackboneConfigMixin

## TimmBackbone [[transformers.TimmBackbone]]

[[autodoc]] models.timm_backbone.TimmBackbone

## TimmBackboneConfig [[transformers.TimmBackboneConfig]]

[[autodoc]] models.timm_backbone.TimmBackboneConfig
