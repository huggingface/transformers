.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Vision Transformer (ViT)
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Vision Transformer (ViT) model was proposed in `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
<https://arxiv.org/abs/2010.11929>`__  by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, 
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. It's the 
first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional
architectures. 


The abstract from the paper is the following:

*While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to 
computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace 
certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not 
necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. 
When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, 
CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while 
requiring substantially fewer computational resources to train.*

Tips:

- To feed images to the Transformer encoder, each image is split into fixed-size patches, which are then linearly embedded. The authors
  also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.
- The Vision Transformer expects each image to be of the same size (resolution), either 224x224 or 384x384 depending on the checkpoint.
  One can use :class:`~transformers.ViTFeatureExtractor` to resize (or rescale) and normalize images for the model. 
- Both the expected image resolution and patch resolution are reflected in the name of each checkpoint. For example, :obj:`google/vit-base-patch16-224`
  refers to a base architecture with image resolution 224x224 and patch resolution of 16x16. All checkpoints can be found on the `hub <https://huggingface.co/models?search=vit>`__.

The original code (written in JAX) can be found `here <https://github.com/google-research/vision_transformer>`__.

Note that we converted the weights from Ross Wightman's `timm library <https://github.com/rwightman/pytorch-image-models>`__, who already converted 
the weights from JAX to PyTorch. Credits go to him!


ViTConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTConfig
    :members:


ViTFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTFeatureExtractor
    :members: 


ViTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTModel
    :members: forward


ViTForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTForImageClassification
    :members: forward