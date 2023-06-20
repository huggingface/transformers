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

# CLIPSeg

## Overview

The CLIPSeg model was proposed in [Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003) by Timo Lüddecke
and Alexander Ecker. CLIPSeg adds a minimal decoder on top of a frozen [CLIP](clip) model for zero- and one-shot image segmentation.

The abstract from the paper is the following:

*Image segmentation is usually addressed by training a
model for a fixed set of object classes. Incorporating additional classes or more complex queries later is expensive
as it requires re-training the model on a dataset that encompasses these expressions. Here we propose a system
that can generate image segmentations based on arbitrary
prompts at test time. A prompt can be either a text or an
image. This approach enables us to create a unified model
(trained once) for three common segmentation tasks, which
come with distinct challenges: referring expression segmentation, zero-shot segmentation and one-shot segmentation.
We build upon the CLIP model as a backbone which we extend with a transformer-based decoder that enables dense
prediction. After training on an extended version of the
PhraseCut dataset, our system generates a binary segmentation map for an image based on a free-text prompt or on
an additional image expressing the query. We analyze different variants of the latter image-based prompts in detail.
This novel hybrid input allows for dynamic adaptation not
only to the three segmentation tasks mentioned above, but
to any binary segmentation task where a text or image query
can be formulated. Finally, we find our system to adapt well
to generalized queries involving affordances or properties*

Tips:

- [`CLIPSegForImageSegmentation`] adds a decoder on top of [`CLIPSegModel`]. The latter is identical to [`CLIPModel`].
- [`CLIPSegForImageSegmentation`] can generate image segmentations based on arbitrary prompts at test time. A prompt can be either a text
(provided to the model as `input_ids`) or an image (provided to the model as `conditional_pixel_values`). One can also provide custom
conditional embeddings (provided to the model as `conditional_embeddings`).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/clipseg_architecture.png"
alt="drawing" width="600"/> 

<small> CLIPSeg overview. Taken from the <a href="https://arxiv.org/abs/2112.10003">original paper.</a> </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/timojl/clipseg).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CLIPSeg. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

<PipelineTag pipeline="image-segmentation"/>

- A notebook that illustrates [zero-shot image segmentation with CLIPSeg](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb).

## CLIPSegConfig

[[autodoc]] CLIPSegConfig
    - from_text_vision_configs

## CLIPSegTextConfig

[[autodoc]] CLIPSegTextConfig

## CLIPSegVisionConfig

[[autodoc]] CLIPSegVisionConfig

## CLIPSegProcessor

[[autodoc]] CLIPSegProcessor

## CLIPSegModel

[[autodoc]] CLIPSegModel
    - forward
    - get_text_features
    - get_image_features

## CLIPSegTextModel

[[autodoc]] CLIPSegTextModel
    - forward

## CLIPSegVisionModel

[[autodoc]] CLIPSegVisionModel
    - forward

## CLIPSegForImageSegmentation

[[autodoc]] CLIPSegForImageSegmentation
    - forward