.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

TrOCR
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TrOCR model was proposed in `TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models
<https://arxiv.org/abs/2109.10282>`__ by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei. TrOCR consists of an image Transformer encoder and an autoregressive text Transformer decoder to
perform `optical character recognition (OCR) <https://en.wikipedia.org/wiki/Optical_character_recognition>`__.

Please refer to the :doc:`VisionEncoderDecoder <visionencoderdecoder>` class on how to use this model.

This model was contributed by `Niels Rogge <https://huggingface.co/nielsr>`__.

The original code can be found `here
<https://github.com/microsoft/unilm/tree/6f60612e7cc86a2a1ae85c47231507a587ab4e01/trocr>`__.


Tips:

- TrOCR is pre-trained in 2 stages before being fine-tuned on downstream datasets. It achieves state-of-the-art results
  on both printed (e.g. the `SROIE dataset <https://paperswithcode.com/dataset/sroie>`__) and handwritten (e.g. the
  `IAM Handwriting dataset <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database>`__) text recognition tasks.
  For more information, see the `official models <https://huggingface.co/models?other=trocr>`__.
- TrOCR is always used within the :doc:`VisionEncoderDecoder <visionencoderdecoder>` framework.

Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TrOCR's :class:`~transformers.VisionEncoderDecoderModel` model accepts images as input and makes use of
:func:`~transformers.generation_utils.GenerationMixin.generate` to autoregressively generate text given the input
image.

The :class:`~transformers.ViTFeatureExtractor` class is responsible for preprocessing the input image and
:class:`~transformers.RobertaTokenizer` decodes the generated target tokens to the target string. The
:class:`~transformers.TrOCRProcessor` wraps :class:`~transformers.ViTFeatureExtractor` and
:class:`~transformers.RobertaTokenizer` into a single instance to both extract the input features and decode the
predicted token ids.

- Step-by-step Optical Character Recognition (OCR)

.. code-block::

        >>> from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        >>> import requests
        >>> from PIL import Image

        >>> processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        >>> # load image from the IAM dataset
        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> pixel_values = processor(image, return_tensors="pt").pixel_values
        >>> generated_ids = model.generate(pixel_values)

        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


See the `model hub <https://huggingface.co/models?filter=trocr>`__ to look for TrOCR checkpoints.


TrOCRConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TrOCRConfig
    :members:


TrOCRProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TrOCRProcessor
    :members: __call__, from_pretrained, save_pretrained, batch_decode, decode, as_target_processor


TrOCRForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TrOCRForCausalLM
    :members: forward
