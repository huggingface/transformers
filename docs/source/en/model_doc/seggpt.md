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

# SegGpt

## Overview

The SegGpt model was proposed in [SegGPT: Segmenting Everything In Context](https://arxiv.org/abs/2304.03284) by Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, Tiejun Huang.
SegGpt is an in-context generalist segmentation model meaning that given an input image, a prompt image and its mask one can obtain related masks in the input image. The model achieves remarkable one-shot results with 56.1 mIoU on COCO-20 and 85.6 mIoU on FSS-1000

The abstract from the paper is the following:

*We present SegGPT, a generalist model for segmenting everything in context. We unify various segmentation tasks into a generalist in-context learning framework that accommodates different kinds of segmentation data by transforming them into the same format of images. The training of SegGPT is formulated as an in-context coloring problem with random color mapping for each data sample. The objective is to accomplish diverse tasks according to the context, rather than relying on specific colors. After training, SegGPT can perform arbitrary segmentation tasks in images or videos via in-context inference, such as object instance, stuff, part, contour, and text. SegGPT is evaluated on a broad range of tasks, including few-shot semantic segmentation, video object segmentation, semantic segmentation, and panoptic segmentation. Our results show strong capabilities in segmenting in-domain and out-of*

Tips:
- One can use [`SegGptImageProcessor`] to prepare image input, prompt and mask to the model.

This model was contributed by [EduardoPacheco](https://huggingface.co/EduardoPacheco).
The original code can be found [here]([<INSERT LINK TO GITHUB REPO HERE>](https://github.com/baaivision/Painter/tree/main)).


## SegGptConfig

[[autodoc]] SegGptConfig

## SegGptImageProcessor

[[autodoc]] SegGptImageProcessor
    - preprocess
    - post_process_masks

## SegGptModel

[[autodoc]] SegGptModel
    - forward
