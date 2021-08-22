.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Vision Transformer (CvT)
-----------------------------------------------------------------------------------------------------------------------

.. note::

    This is a recently introduced model so the API hasn't been tested extensively. There may be some bugs or slight
    breaking changes to fix it in the future. If you see something strange, file a `Github Issue
    <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__.


Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Convolutional Vision Transformer (CvT) model was proposed in `CvT: Introducing Convolutions to Vision Transformers
<https://arxiv.org/abs/2103.15808>`__ by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei
Zhang.

The abstract from the paper is the following:

*We present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision
Transformer (ViT) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs.
This is accomplished through two primary modifications: a hierarchy of Transformers containing a new convolutional
token embedding, and a convolutional Transformer block leveraging a convolutional projection. These changes introduce
desirable properties of convolutional neural networks (CNNs) to the ViT architecture (\ie shift, scale, and distortion
invariance) while maintaining the merits of Transformers (\ie dynamic attention, global context, and better
generalization). We validate CvT by conducting extensive experiments, showing that this approach achieves
state-of-the-art performance over other Vision Transformers and ResNets on ImageNet-1k, with fewer parameters and lower
FLOPs. In addition, performance gains are maintained when pretrained on larger datasets (\eg ImageNet-22k) and
fine-tuned to downstream tasks. Pre-trained on ImageNet-22k, our CvT-W24 obtains a top-1 accuracy of 87.7\% on the
ImageNet-1k val set. Finally, our results show that the positional encoding, a crucial component in existing Vision
Transformers, can be safely removed in our model, simplifying the design for higher resolution vision tasks.*

This model was contributed by `ANugunjNaman <https://github.com/ANugunjNaman>`__. The original code (written in JAX)
can be found `here <https://github.com/microsoft/CvT>`__.

CvTConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CvTConfig
    :members:


CvTFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CvTFeatureExtractor
    :members: __call__


CvTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CvTModel
    :members: forward


CvTForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CvTForImageClassification
    :members: forward
