<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License. 

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.


-->

# SuperPoint

## Overview

The SuperPoint model was proposed
in [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629) by Daniel
DeTone, Tomasz Malisiewicz and Andrew Rabinovich.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*This paper presents a self-supervised framework for training interest point detectors and descriptors suitable for a
large number of multiple-view geometry problems in computer vision. As opposed to patch-based neural networks, our
fully-convolutional model operates on full-sized images and jointly computes pixel-level interest point locations and
associated descriptors in one forward pass. We introduce Homographic Adaptation, a multi-scale, multi-homography
approach for boosting interest point detection repeatability and performing cross-domain adaptation (e.g.,
synthetic-to-real). Our model, when trained on the MS-COCO generic image dataset using Homographic Adaptation, is able
to repeatedly detect a much richer set of interest points than the initial pre-adapted deep model and any other
traditional corner detector. The final system gives rise to state-of-the-art homography estimation results on HPatches
when compared to LIFT, SIFT and ORB.*

Tips:

- See the code examples below each model regarding usage.

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/magicleap/SuperPointPretrainedNetwork).

## SuperPointConfig

[[autodoc]] SuperPointConfig

## SuperPointImageProcessor

[[autodoc]] SuperPointImageProcessor 
    - preprocess

## SuperPointModel

[[autodoc]] SuperPointModel
    - forward

## SuperPointForInterestPointDescription

[[autodoc]] SuperPointForInterestPointDescription
    - forward

