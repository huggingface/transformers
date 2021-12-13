.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Vision Encoder Decoder Models
-----------------------------------------------------------------------------------------------------------------------

The :class:`~transformers.VisionEncoderDecoderModel` can be used to initialize an image-to-text-sequence model with any
pretrained vision autoencoding model as the encoder (*e.g.* :doc:`ViT <vit>`, :doc:`BEiT <beit>`, :doc:`DeiT <deit>`)
and any pretrained language model as the decoder (*e.g.* :doc:`RoBERTa <roberta>`, :doc:`GPT2 <gpt2>`, :doc:`BERT
<bert>`).

The effectiveness of initializing image-to-text-sequence models with pretrained checkpoints has been shown in (for
example) `TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models
<https://arxiv.org/abs/2109.10282>`__ by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei.

An example of how to use a :class:`~transformers.VisionEncoderDecoderModel` for inference can be seen in :doc:`TrOCR
<trocr>`.


VisionEncoderDecoderConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisionEncoderDecoderConfig
    :members:


VisionEncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.VisionEncoderDecoderModel
    :members: forward, from_encoder_decoder_pretrained


FlaxVisionEncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxVisionEncoderDecoderModel
    :members: __call__, from_encoder_decoder_pretrained
