<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# YOLOS

## Overview

The YOLOS model was proposed in [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/abs/2106.00666) by Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu.
YOLOS proposes to just leverage the plain [Vision Transformer (ViT)](vit) for object detection, inspired by DETR. It turns out that a base-sized encoder-only Transformer can also achieve 42 AP on COCO, similar to DETR and much more complex frameworks such as Faster R-CNN.

The abstract from the paper is the following:

*Can Transformer perform 2D object- and region-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the 2D spatial structure? To answer this question, we present You Only Look at One Sequence (YOLOS), a series of object detection models based on the vanilla Vision Transformer with the fewest possible modifications, region priors, as well as inductive biases of the target task. We find that YOLOS pre-trained on the mid-sized ImageNet-1k dataset only can already achieve quite competitive performance on the challenging COCO object detection benchmark, e.g., YOLOS-Base directly adopted from BERT-Base architecture can obtain 42.0 box AP on COCO val. We also discuss the impacts as well as limitations of current pre-train schemes and model scaling strategies for Transformer in vision through YOLOS.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yolos_architecture.png"
alt="drawing" width="600"/>

<small> YOLOS architecture. Taken from the <a href="https://arxiv.org/abs/2106.00666">original paper</a>.</small>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/hustvl/YOLOS).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with YOLOS.

<PipelineTag pipeline="object-detection"/>

- All example notebooks illustrating inference + fine-tuning [`YolosForObjectDetection`] on a custom dataset can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/YOLOS).
- See also: [Object detection task guide](../tasks/object_detection)

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<Tip>

Use [`YolosImageProcessor`] for preparing images (and optional targets) for the model. Contrary to [DETR](detr), YOLOS doesn't require a `pixel_mask` to be created.

</Tip>

## YolosConfig

[[autodoc]] YolosConfig

## YolosImageProcessor

[[autodoc]] YolosImageProcessor
    - preprocess
    - pad
    - post_process_object_detection

## YolosFeatureExtractor

[[autodoc]] YolosFeatureExtractor
    - __call__
    - pad
    - post_process_object_detection

## YolosModel

[[autodoc]] YolosModel
    - forward

## YolosForObjectDetection

[[autodoc]] YolosForObjectDetection
    - forward
