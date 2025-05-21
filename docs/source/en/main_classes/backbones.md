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

# Backbone

A backbone is a model used for feature extraction for higher level computer vision tasks such as object detection and image classification. Transformers provides an [`AutoBackbone`] class for initializing a Transformers backbone from pretrained model weights, and two utility classes:

* [`~utils.BackboneMixin`] enables initializing a backbone from Transformers or [timm](https://hf.co/docs/timm/index) and includes functions for returning the output features and indices.
* [`~utils.BackboneConfigMixin`] sets the output features and indices of the backbone configuration.

[timm](https://hf.co/docs/timm/index) models are loaded with the [`TimmBackbone`] and [`TimmBackboneConfig`] classes.

Backbones are supported for the following models:

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

## AutoBackbone

[[autodoc]] AutoBackbone

## BackboneMixin

[[autodoc]] utils.BackboneMixin

## BackboneConfigMixin

[[autodoc]] utils.BackboneConfigMixin

## TimmBackbone

[[autodoc]] models.timm_backbone.TimmBackbone

## TimmBackboneConfig

[[autodoc]] models.timm_backbone.TimmBackboneConfig
