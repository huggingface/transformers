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

# SigLIP

## Overview

The SigLIP model was proposed in [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) by Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. SigLIP proposes to replace the loss function used in [CLIP](clip) by a simple pairwise sigmoid loss. This results in better performance in terms of zero-shot classification accuracy on ImageNet.

The abstract from the paper is the following:

*We propose a simple pairwise Sigmoid loss for Language-Image Pre-training (SigLIP). Unlike standard contrastive learning with softmax normalization, the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization. The sigmoid loss simultaneously allows further scaling up the batch size, while also performing better at smaller batch sizes. Combined with Locked-image Tuning, with only four TPUv4 chips, we train a SigLiT model that achieves 84.5% ImageNet zero-shot accuracy in two days. The disentanglement of the batch size from the loss further allows us to study the impact of examples vs pairs and negative to positive ratio. Finally, we push the batch size to the extreme, up to one million, and find that the benefits of growing batch size quickly diminish, with a more reasonable batch size of 32k being sufficient.*

Tips:

- Usage of SigLIP is identical to [CLIP](clip). The only difference is the training loss, which does not require a global view of all the pairwise similarities of images and texts within a batch. 

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/google-research/big_vision/tree/main).


## SiglipConfig

[[autodoc]] SiglipConfig
    - from_text_vision_configs

## SiglipTextConfig

[[autodoc]] SiglipTextConfig

## SiglipVisionConfig

[[autodoc]] SiglipVisionConfig

## SiglipTokenizer

[[autodoc]] SiglipTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SiglipImageProcessor

[[autodoc]] SiglipImageProcessor
    - preprocess

## SiglipProcessor

[[autodoc]] SiglipProcessor

## SiglipModel

[[autodoc]] SiglipModel
    - forward
    - get_text_features
    - get_image_features

## SiglipTextModel

[[autodoc]] SiglipTextModel
    - forward


## SiglipVisionModel

[[autodoc]] SiglipVisionModel
    - forward
