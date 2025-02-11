<!--Copyright 2022 NVIDIA and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GroupViT

## Overview

The GroupViT model was proposed in [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094) by Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.
Inspired by [CLIP](clip), GroupViT is a vision-language model that can perform zero-shot semantic segmentation on any given vocabulary categories.

The abstract from the paper is the following:

*Grouping and recognition are important components of visual scene understanding, e.g., for object detection and semantic segmentation. With end-to-end deep learning systems, grouping of image regions usually happens implicitly via top-down supervision from pixel-level recognition labels. Instead, in this paper, we propose to bring back the grouping mechanism into deep networks, which allows semantic segments to emerge automatically with only text supervision. We propose a hierarchical Grouping Vision Transformer (GroupViT), which goes beyond the regular grid structure representation and learns to group image regions into progressively larger arbitrary-shaped segments. We train GroupViT jointly with a text encoder on a large-scale image-text dataset via contrastive losses. With only text supervision and without any pixel-level annotations, GroupViT learns to group together semantic regions and successfully transfers to the task of semantic segmentation in a zero-shot manner, i.e., without any further fine-tuning. It achieves a zero-shot accuracy of 52.3% mIoU on the PASCAL VOC 2012 and 22.4% mIoU on PASCAL Context datasets, and performs competitively to state-of-the-art transfer-learning methods requiring greater levels of supervision.*

This model was contributed by [xvjiarui](https://huggingface.co/xvjiarui). The TensorFlow version was contributed by [ariG23498](https://huggingface.co/ariG23498) with the help of [Yih-Dar SHIEH](https://huggingface.co/ydshieh), [Amy Roberts](https://huggingface.co/amyeroberts), and [Joao Gante](https://huggingface.co/joaogante).
The original code can be found [here](https://github.com/NVlabs/GroupViT).

## Usage tips
 
- You may specify `output_segmentation=True` in the forward of `GroupViTModel` to get the segmentation logits of input texts. 

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with GroupViT.

- The quickest way to get started with GroupViT is by checking the [example notebooks](https://github.com/xvjiarui/GroupViT/blob/main/demo/GroupViT_hf_inference_notebook.ipynb) (which showcase zero-shot segmentation inference).
- One can also check out the [HuggingFace Spaces demo](https://huggingface.co/spaces/xvjiarui/GroupViT) to play with GroupViT. 

## GroupViTConfig

[[autodoc]] GroupViTConfig
    - from_text_vision_configs

## GroupViTTextConfig

[[autodoc]] GroupViTTextConfig

## GroupViTVisionConfig

[[autodoc]] GroupViTVisionConfig

<frameworkcontent>
<pt>

## GroupViTModel

[[autodoc]] GroupViTModel
    - forward
    - get_text_features
    - get_image_features

## GroupViTTextModel

[[autodoc]] GroupViTTextModel
    - forward

## GroupViTVisionModel

[[autodoc]] GroupViTVisionModel
    - forward

</pt>
<tf>

## TFGroupViTModel

[[autodoc]] TFGroupViTModel
    - call
    - get_text_features
    - get_image_features

## TFGroupViTTextModel

[[autodoc]] TFGroupViTTextModel
    - call

## TFGroupViTVisionModel

[[autodoc]] TFGroupViTVisionModel
    - call

</tf>
</frameworkcontent>
