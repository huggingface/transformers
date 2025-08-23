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

# Conditional DETR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Conditional DETR model was proposed in [Conditional DETR for Fast Training Convergence](https://huggingface.co/papers/2108.06152) by Depu Meng, Xiaokang Chen, Zejia Fan, Gang Zeng, Houqiang Li, Yuhui Yuan, Lei Sun, Jingdong Wang. Conditional DETR presents a conditional cross-attention mechanism for fast DETR training. Conditional DETR converges 6.7× to 10× faster than DETR.

The abstract from the paper is the following:

*The recently-developed DETR approach applies the transformer encoder and decoder architecture to object detection and achieves promising performance. In this paper, we handle the critical issue, slow training convergence, and present a conditional cross-attention mechanism for fast DETR training. Our approach is motivated by that the cross-attention in DETR relies highly on the content embeddings for localizing the four extremities and predicting the box, which increases the need for high-quality content embeddings and thus the training difficulty. Our approach, named conditional DETR, learns a conditional spatial query from the decoder embedding for decoder multi-head cross-attention. The benefit is that through the conditional spatial query, each cross-attention head is able to attend to a band containing a distinct region, e.g., one object extremity or a region inside the object box. This narrows down the spatial range for localizing the distinct regions for object classification and box regression, thus relaxing the dependence on the content embeddings and easing the training. Empirical results show that conditional DETR converges 6.7× faster for the backbones R50 and R101 and 10× faster for stronger backbones DC5-R50 and DC5-R101. Code is available at https://github.com/Atten4Vis/ConditionalDETR.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/conditional_detr_curve.jpg"
alt="drawing" width="600"/>

<small> Conditional DETR shows much faster convergence compared to the original DETR. Taken from the <a href="https://huggingface.co/papers/2108.06152">original paper</a>.</small>

This model was contributed by [DepuMeng](https://huggingface.co/DepuMeng). The original code can be found [here](https://github.com/Atten4Vis/ConditionalDETR).

## Resources

- Scripts for finetuning [`ConditionalDetrForObjectDetection`] with [`Trainer`] or [Accelerate](https://huggingface.co/docs/accelerate/index) can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- See also: [Object detection task guide](../tasks/object_detection).

## ConditionalDetrConfig

[[autodoc]] ConditionalDetrConfig

## ConditionalDetrImageProcessor

[[autodoc]] ConditionalDetrImageProcessor
    - preprocess

## ConditionalDetrImageProcessorFast

[[autodoc]] ConditionalDetrImageProcessorFast
    - preprocess
    - post_process_object_detection
    - post_process_instance_segmentation
    - post_process_semantic_segmentation
    - post_process_panoptic_segmentation

## ConditionalDetrFeatureExtractor

[[autodoc]] ConditionalDetrFeatureExtractor
    - __call__
    - post_process_object_detection
    - post_process_instance_segmentation
    - post_process_semantic_segmentation
    - post_process_panoptic_segmentation

## ConditionalDetrModel

[[autodoc]] ConditionalDetrModel
    - forward

## ConditionalDetrForObjectDetection

[[autodoc]] ConditionalDetrForObjectDetection
    - forward

## ConditionalDetrForSegmentation

[[autodoc]] ConditionalDetrForSegmentation
    - forward
