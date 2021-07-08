.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

LayoutLMV2
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LayoutLMV2 model was proposed in `LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding
<https://arxiv.org/abs/2012.14740>`__ by Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu,
Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMV2 improves `LayoutLM
<https://huggingface.co/transformers/model_doc/layoutlm.html>`__ to obtain state-of-the-art results across several
document image understanding benchmarks:

- information extraction from scanned documents: the `FUNSD <https://guillaumejaume.github.io/FUNSD/>`__ dataset (a
  collection of 199 annotated forms comprising more than 30,000 words), the `CORD <https://github.com/clovaai/cord>`__
  dataset (a collection of 800 receipts for training, 100 for validation and 100 for testing), the `SROIE
  <https://rrc.cvc.uab.es/?ch=13>`__ dataset (a collection of 626 receipts for training and 347 receipts for testing)
  and the `Kleister-NDA <https://github.com/applicaai/kleister-nda>`__ dataset (a collection of non-disclosure
  agreements from the EDGAR database, including 254 documents for training, 83 documents for validation, and 203
  documents for testing).
- document image classification: the `RVL-CDIP <https://www.cs.cmu.edu/~aharley/rvl-cdip/>`__ dataset (a collection of
  400,000 images belonging to one of 16 classes).

The abstract from the paper is the following:

*Pre-training of text and layout has proved effective in a variety of visually-rich document understanding tasks due to
its effective model architecture and the advantage of large-scale unlabeled scanned/digital-born documents. In this
paper, we present LayoutLMv2 by pre-training text, layout and image in a multi-modal framework, where new model
architectures and pre-training tasks are leveraged. Specifically, LayoutLMv2 not only uses the existing masked
visual-language modeling task but also the new text-image alignment and text-image matching tasks in the pre-training
stage, where cross-modality interaction is better learned. Meanwhile, it also integrates a spatial-aware self-attention
mechanism into the Transformer architecture, so that the model can fully understand the relative positional
relationship among different text blocks. Experiment results show that LayoutLMv2 outperforms strong baselines and
achieves new state-of-the-art results on a wide variety of downstream visually-rich document understanding tasks,
including FUNSD (0.7895 -> 0.8420), CORD (0.9493 -> 0.9601), SROIE (0.9524 -> 0.9781), Kleister-NDA (0.834 -> 0.852),
RVL-CDIP (0.9443 -> 0.9564), and DocVQA (0.7295 -> 0.8672). The pre-trained LayoutLMv2 model is publicly available at
this https URL.*

Tips:

- LayoutLMv2 uses Facebook AI's `Detectron2 <https://github.com/facebookresearch/detectron2/>`__ package for its visual
  backbone. See `this link <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`__ for installation
  instructions.
- In addition to :obj:`input_ids`, :meth:`~transformer.LayoutLMv2Model.forward` expects 2 additional inputs, namely
  :obj:`image` and :obj:`bbox`. The :obj:`image` input corresponds to the original document image in which the text
  tokens occur. The model expects each document image to be of size 224x224. This means that if you have a batch of
  document images, :obj:`image` should be a tensor of shape (batch_size, 3, 224, 224). This can be either a
  :obj:`torch.Tensor` or a :obj:`Detectron2.structures.ImageList`. You don't need to normalize the channels, as this is
  done by the model. The :obj:`bbox` input are the bounding boxes (i.e. 2D-positions) of the input text tokens. This is
  identical to :class:`~transformer.LayoutLMModel`. These can be obtained using an external OCR engine such as
  Google's `Tesseract <https://github.com/tesseract-ocr/tesseract>`__ (there's a `Python wrapper
  <https://pypi.org/project/pytesseract/>`__ available). Each bounding box should be in (x0, y0, x1, y1) format, where
  (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the
  position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000
  scale. To normalize, you can use the following function:

.. code-block::

    def normalize_bbox(bbox, width, height):
         return [
             int(1000 * (bbox[0] / width)),
             int(1000 * (bbox[1] / height)),
             int(1000 * (bbox[2] / width)),
             int(1000 * (bbox[3] / height)),
         ]

Here, :obj:`width` and :obj:`height` correspond to the width and height of the original document in which the token
occurs. Those can be obtained using the Python Image Library (PIL) library for example, as follows:

.. code-block::

    from PIL import Image

    image = Image.open("name_of_your_document - can be a png file, pdf, etc.")

    width, height = image.size

- Internally, :class:`~transformer.LayoutLMv2Model` will send the :obj:`image` input through its visual backbone to
  obtain a lower-resolution feature map, whose shape is equal to the :obj:`image_feature_pool_shape` attribute of
  :class:`~transformer.LayoutLMv2Config`. This feature map is then flattened to obtain a sequence of image tokens. As
  the size of the feature map is 7x7 by default, one obtains 49 image tokens. These are then concatenated with the text
  tokens, and send through the Transformer encoder. This means that the last hidden states of the model will have a
  length of 512 + 49 = 561, if you pad the text tokens up to the max length.
- When calling :meth:`~transformer.LayoutLMv2Model.from_pretrained`, a warning will be printed with a long list of 
  parameter names that are not initialized. This is not a problem, as these parameters are batch normalization statistics,
  which are going to have values when fine-tuning on a custom dataset.

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__. The original code can be found `here
<https://github.com/microsoft/unilm/tree/master/layoutlmv2>`__.

LayoutLMv2Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2Config
    :members:


LayoutLMv2Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2Tokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


LayoutLMv2TokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2TokenizerFast
    :members:


LayoutLMv2Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2Model
    :members: forward


LayoutLMv2ForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2ForSequenceClassification
    :members:


LayoutLMv2ForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2ForTokenClassification
    :members:


LayoutLMv2ForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2ForQuestionAnswering
    :members:
