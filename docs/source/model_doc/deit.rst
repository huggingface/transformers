.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

DeiT
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DeiT model was proposed in `Training data-efficient image transformers & distillation through attention
<https://arxiv.org/abs/2012.12877>`__  by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, 
Alexandre Sablayrolles, Hervé Jégou. The Vision Transformer (ViT) introduced in `Dosovitskiy et al., 2020 <https://arxiv.org/abs/2010.11929>`__ 
has shown that one can match or even outperform existing convolutional neural networks using a Transformer encoder (BERT-like).
However, the ViT models introduced in that paper required training on expensive infrastructure for multiple weeks, using external data. 
DeiT (data-efficient image transformers) are more efficiently trained transformers for image classification, requiring far less data and 
far less computing resources compared to the original ViT models.  

The abstract from the paper is the following:

*Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. 
However, these visual transformers are pre-trained with hundreds of millions of images using an expensive infrastructure, thereby 
limiting their adoption. In this work, we produce a competitive convolution-free transformer by training on Imagenet only. We train 
them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% 
(single-crop evaluation) on ImageNet with no external data. More importantly, we introduce a teacher-student strategy specific to 
transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the 
interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive 
with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and 
models.*

Tips:

- The authors of DeiT released two flavours of checkpoints.
- The first type of DeiT checkpoints are more efficiently trained ViT models, which you can directly plug into :class:`~transformers.ViTModel` 
  or :class:`~transformers.ViTForImageClassification`. Techniques like data augmentation, optimization, and regularization were used in order 
  to simulate training on a much larger dataset (while only using ImageNet-1k for pre-training). There are 4 variants available (in 3 different 
  sizes): `facebook/deit-tiny-patch16-224`, `facebook/deit-small-patch16-224`, `facebook/deit-base-patch16-224` and `facebook/deit-base-patch16-384`. 
- The second type of DeiT checkpoints are distillated versions, which use a so-called distillation token to effectively learn from
  a pre-trained teacher (which, in the DeiT paper, is a ResNet like-model). These second type of models can be pluged into :class:`~transformers.DeiTModel`,
  :class:`~transformers.DeiTForImageClassification` or :class:`~transformers.DeiTForImageClassificationWithTeacher`. These add a special distillation token 
  besides the class token ([CLS]) to the patch sequences, and place a classification head on top of both the final hidden state of the class and distillation 
  tokens. The former is trained using regular cross-entropy between the prediction of the [CLS] head and the ground-truth label, while the latter is trained 
  using hard distillation (cross-entropy  between the prediction of the distillation head and the label predicted by the teacher). The distillation token 
  is learned through backpropagation, by interacting with the class and patch tokens through the self-attention layers. At inference time, one takes the 
  average prediction between both heads as final prediction. Note that the authors also did try soft distillation, but hard distillation gave the best results. 
- All released checkpoints were pre-trained and fine-tuned on ImageNet-1k only. No external data was used. This is in contrast with the original
  ViT model, which used external data like the JFT-300M dataset and Imagenet-21k for pre-training.


DeiTConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DeiTConfig
    :members:


DeiTFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DeiTFeatureExtractor
    :members: __call__


DeiTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DeiTModel
    :members: forward


DeiTForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DeiTForImageClassification
    :members: forward


DeiTForImageClassificationWithTeacher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DeiTForImageClassificationWithTeacher
    :members: forward