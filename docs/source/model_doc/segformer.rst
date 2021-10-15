.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

SegFormer
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SegFormer model was proposed in `SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
<https://arxiv.org/abs/2105.15203>`__ by Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping
Luo. The model consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great
results on image segmentation benchmarks such as ADE20K and Cityscapes.

The abstract from the paper is the following:

*We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with
lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel
hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding,
thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution
differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from
different layers, and thus combining both local attention and global attention to render powerful representations. We
show that this simple and lightweight design is the key to efficient segmentation on Transformers. We scale our
approach up to obtain a series of models from SegFormer-B0 to SegFormer-B5, reaching significantly better performance
and efficiency than previous counterparts. For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters,
being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on
Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C.*

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__. The original code can be found `here
<https://github.com/NVlabs/SegFormer>`__.

SegformerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerConfig
    :members:


SegformerFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerFeatureExtractor
    :members: __call__


SegformerModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerModel
    :members: forward


SegformerDecodeHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerDecodeHead
    :members: forward


SegformerForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerForImageClassification
    :members: forward


SegformerForSemanticSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerForSemanticSegmentation
    :members: forward
