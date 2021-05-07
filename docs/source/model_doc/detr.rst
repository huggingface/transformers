.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

DETR
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DETR model was proposed in `End-to-End Object Detection with Transformers <https://arxiv.org/abs/2005.12872>`__ by
Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov and Sergey Zagoruyko. DETR
consists of a convolutional backbone followed by an encoder-decoder Transformer which can be trained end-to-end for
object detection. It greatly simplifies a lot of the complexity of models like Faster-R-CNN and Mask-R-CNN, which use
things like region proposals, non-maximum suppression procedure and anchor generation. Moreover, DETR can also be
naturally extended to perform panoptic segmentation, by simply adding a mask head on top of the decoder outputs.

The abstract from the paper is the following:

*We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the
detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression
procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the
new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via
bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries,
DETR reasons about the relations of the objects and the global image context to directly output the final set of
predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many
other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and
highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily
generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive
baselines.*

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__. The original code can be found `here
<https://github.com/facebookresearch/detr>`__.

Here's a TLDR explaining how the model works:

First, an image is sent through a pre-trained convolutional backbone (in the paper, the authors use
ResNet-50/ResNet-101). Let's assume we also add a batch dimension. This means that the input to the backbone is a
tensor of shape :obj:`(batch_size, 3, height, width)`, assuming the image has 3 color channels (RGB). The CNN backbone
outputs a new lower-resolution feature map, typically of shape :obj:`(batch_size, 2048, height/32, width/32)`. This is
then projected to match the hidden dimension of the Transformer of DETR, which is :obj:`256` by default, using a
:obj:`nn.Conv2D` layer. So now, we have a tensor of shape :obj:`(batch_size, 256, height/32, width/32).` Next, the
feature map is flattened and transposed to obtain a tensor of shape :obj:`(batch_size, seq_len, d_model)` =
:obj:`(batch, width/32*height/32, 256)`. So a difference with NLP models is that the sequence length is actually longer
than usual, but with a smaller :obj:`d_model` (which in NLP is typically 768 or higher).

Next, this is sent through the encoder, outputting :obj:`encoder_hidden_states` of the same shape (you can consider
these as image features). Next, so-called **object queries** are sent through the decoder. This is a tensor of shape
:obj:`(batch_size, num_queries, d_model)`, with :obj:`num_queries` typically set to 100 and initialized with zeros.
These input embeddings are learnt positional encodings that the authors refer to as object queries, and similarly to the 
encoder, they are added to the input of each attention layer. Each object query will look for a particular object in the 
image. The decoder updates these embeddings through multiple self-attention and encoder-decoder attention layers to output 
:obj:`decoder_hidden_states` of the same shape: :obj:`(batch_size, num_queries, d_model)`. Next, two heads are added on top 
for object detection: a linear layer for classifying each object query into one of the objects or "no object", and a MLP to 
predict bounding boxes for each query.

The model is trained using a **bipartite matching loss**: so what we actually do is compare the predicted classes +
bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N
(so if an image only contains 4 objects, 96 annotations will just have a "no object" as class and "no bounding box" as
bounding box). The `Hungarian matching algorithm <https://en.wikipedia.org/wiki/Hungarian_algorithm>`__ is used to
create a one-to-one mapping of each of the N queries to each of the N annotations. Next, standard cross-entropy and L1
bounding box losses are used to optimize the parameters of the model.

Tips:

- DETR uses so-called **object queries** to detect objects in an image. The number of queries determines the maximum
  number of objects that can be detected in a single image, and is set to 100 by default (see parameter
  :obj:`num_queries` of :class:`~transformers.DetrConfig`). Note that it's good to have some slack (in COCO, the
  authors used 100, while the maximum number of objects in a COCO image is ~70).
- The decoder of DETR updates the query embeddings in parallel. This is different from language models like GPT-2,
  which use autoregressive decoding instead of parallel. Hence, no causal attention mask is used.
- DETR adds position embeddings to the hidden states at each self-attention and cross-attention layer before projecting
  to queries and keys. For the position embeddings of the image, one can choose between fixed sinusoidal or learned
  absolute position embeddings. By default, the parameter :obj:`position_embedding_type` of
  :class:`~transformers.DetrConfig` is set to :obj:`"sine"`.
- During training, the authors of DETR did find it helpful to use auxiliary losses in the decoder, especially to help
  the model output the correct number of objects of each class. If you set the parameter :obj:`auxiliary_loss` of
  :class:`~transformers.DetrConfig` to :obj:`True`, then prediction feedforward neural networks and Hungarian losses
  are added after each decoder layer (with the FFNs sharing parameters).
- If you want to train the model in a distributed environment across multiple nodes, then one should update the
  `num_boxes` variable in the `SetCriterion` class of `modeling_detr.py`. When training on multiple nodes, this should
  be set to the average number of target boxes across all nodes, as can be seen in the original implementation `here
  <https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L227-L232>`__.
- :class:`~transformers.DetrForObjectDetection` can be initialized with any convolutional backbone available in the `timm 
  library <https://github.com/rwightman/pytorch-image-models>`__. Initializing with a MobileNet backbone for example can be 
  done by setting the :obj:`backbone` attribute of :class:`~transformers.DetrConfig` to :obj:`"tf_mobilenetv3_small_075"`, 
  and then initializing :class:`~transformers.DetrForObjectDetection` with that config.
- At inference time, DETR resizes the input images such that the shortest side is at least 800 pixels while the longest at most 
  1333 pixels. One can use :class:`~transformers.DetrFeatureExtractor` to prepare images (and optional annotations in COCO format) 
  for the model. Due to this, images in a batch can have different sizes. DETR solves this by padding images up to the largest 
  size in a batch, and by creating a pixel mask that indicates which pixels are real/which are padding. Alternatively, one can also 
  define a custom :obj:`collate_fn` in order to batch images together, using :meth:`~transformers.DetrFeatureExtractor.pad_and_create_pixel_mask`. 

DetrConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DetrConfig
    :members:


DetrFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DetrFeatureExtractor
    :members: __call__, pad_and_create_pixel_mask


DetrModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DetrModel
    :members: forward


DetrForObjectDetection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DetrForObjectDetection
    :members: forward


DetrForSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DetrForSegmentation
    :members: forward
