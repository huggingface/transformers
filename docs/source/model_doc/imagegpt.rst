.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

ImageGPT
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ImageGPT model was proposed in `Generative Pretraining from Pixels <https://openai.com/blog/image-gpt/>`__ by Mark
Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever. ImageGPT (iGPT) is a GPT-2-like
model trained to predict the next pixel value, allowing for both unconditional and conditional image generation.

The abstract from the paper is the following:

*Inspired by progress in unsupervised representation learning for natural language, we examine whether similar models
can learn useful representations for images. We train a sequence Transformer to auto-regressively predict pixels,
without incorporating knowledge of the 2D input structure. Despite training on low-resolution ImageNet without labels,
we find that a GPT-2 scale model learns strong image representations as measured by linear probing, fine-tuning, and
low-data classification. On CIFAR-10, we achieve 96.3% accuracy with a linear probe, outperforming a supervised Wide
ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models. We are also
competitive with self-supervised benchmarks on ImageNet when substituting pixels for a VQVAE encoding, achieving 69.0%
top-1 accuracy on a linear probe of our features.*

The figure below summarizes the approach (taken from the `original paper
<https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf>`__):

.. image:: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/imagegpt_architecture.png
  :width: 600

Tips:

- ImageGPT is almost exactly the same as :doc:`GPT-2 <gpt2>`, with the exception that a different activation function
  is used (namely "quick gelu"), and the layer normalization layers don't mean center the inputs. ImageGPT also doesn't
  have tied input- and output embeddings.
- As the time- and memory requirements of the attention mechanism of Transformers scales quadratically in the sequence
  length, the authors pre-trained ImageGPT on smaller input resolutions, such as 32x32 and 64x64. However, feeding a
  sequence of 32x32x3=3072 tokens from 0..255 into a Transformer is still prohibitively large. Therefore, the authors
  applied k-means clustering to the (R,G,B) pixel values with k=512. This way, we only have a 32*32 = 1024-long
  sequence, but now of integers in the range 0..511. So we are shrinking the sequence length at the cost of a bigger
  embedding matrix. In other words, the vocabulary size of ImageGPT is 512, + 1 for a special "start of sentence" (SOS)
  token, used at the beginning of every sequence. One can use :class:`~transformers.ImageGPTFeatureExtractor` to
  prepare images for the model.
- Despite being pre-trained entirely unsupervised (i.e. without the use of any labels), ImageGPT produces fairly
  performant image features useful for downstream tasks, such as image classification. The authors showed that the
  features in the middle of the network are the most performant, and can be used as-is to train a linear model (such as
  a sklearn logistic regression model for example). This is also referred to as "linear probing". Features can be
  easily obtained by first forwarding the image through the model, then specifying `output_hidden_states=True`, and
  then average-pool the hidden states at whatever layer you like.
- Alternatively, one can further fine-tune the entire model on a downstream dataset, similar to BERT. For this, you can
  use :class:`~transformers.ImageGPTForImageClassification`.
- ImageGPT comes in different sizes: there's ImageGPT-small, ImageGPT-medium and ImageGPT-large. The authors did also
  train an XL variant, which they didn't release. The differences in size are summarized in the following table:

+-------------------+----------------------+-----------------+---------------------+--------------+
| **Model variant** | **Number of layers** | **Hidden size** | **Number of heads** | **# params** |
+-------------------+----------------------+-----------------+---------------------+--------------+
| iGPT-small        | 24                   | 512             | 8                   | 76 million   |
+-------------------+----------------------+-----------------+---------------------+--------------+
| iGPT-medium       | 36                   | 1024            | 8                   | 455 million  |
+-------------------+----------------------+-----------------+---------------------+--------------+
| iGPT-large        | 48                   | 1536            | 16                  | 1.4 million  |
+-------------------+----------------------+-----------------+---------------------+--------------+
| iGPT-XL           | 60                   | 3072            | not specified       | 6.8 billion  |
+-------------------+----------------------+-----------------+---------------------+--------------+

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__, based on `this issue
<https://github.com/openai/image-gpt/issues/7>`__. The original code can be found `here
<https://github.com/openai/image-gpt>`__.

ImageGPTConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ImageGPTConfig
    :members:

ImageGPTFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ImageGPTFeatureExtractor
    :members: __call__

ImageGPTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ImageGPTModel
    :members: forward


ImageGPTForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ImageGPTForCausalLM
    :members: forward


ImageGPTForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ImageGPTForImageClassification
    :members: forward
