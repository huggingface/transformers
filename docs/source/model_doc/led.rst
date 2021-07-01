.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

LED
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LED model was proposed in `Longformer: The Long-Document Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz
Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales
quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention
mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or
longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local
windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we
evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In
contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our
pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on
WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting
long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization
dataset.*

Tips:

- :class:`~transformers.LEDForConditionalGeneration` is an extension of
  :class:`~transformers.BartForConditionalGeneration` exchanging the traditional *self-attention* layer with
  *Longformer*'s *chunked self-attention* layer. :class:`~transformers.LEDTokenizer` is an alias of
  :class:`~transformers.BartTokenizer`.
- LED works very well on long-range *sequence-to-sequence* tasks where the ``input_ids`` largely exceed a length of
  1024 tokens.
- LED pads the ``input_ids`` to be a multiple of ``config.attention_window`` if required. Therefore a small speed-up is
  gained, when :class:`~transformers.LEDTokenizer` is used with the ``pad_to_multiple_of`` argument.
- LED makes use of *global attention* by means of the ``global_attention_mask`` (see
  :class:`~transformers.LongformerModel`). For summarization, it is advised to put *global attention* only on the first
  ``<s>`` token. For question answering, it is advised to put *global attention* on all tokens of the question.
- To fine-tune LED on all 16384, it is necessary to enable *gradient checkpointing* by setting
  ``config.gradient_checkpointing = True``.
- A notebook showing how to evaluate LED, can be accessed `here
  <https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing>`__.
- A notebook showing how to fine-tune LED, can be accessed `here
  <https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing>`__.

This model was contributed by `patrickvonplaten <https://huggingface.co/patrickvonplaten>`__.


LEDConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LEDConfig
    :members:


LEDTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LEDTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


LEDTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LEDTokenizerFast
    :members:


LED specific outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.models.led.modeling_led.LEDEncoderBaseModelOutput
    :members: 

.. autoclass:: transformers.models.led.modeling_led.LEDSeq2SeqModelOutput
    :members: 

.. autoclass:: transformers.models.led.modeling_led.LEDSeq2SeqLMOutput
    :members: 

.. autoclass:: transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput
    :members: 

.. autoclass:: transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput
    :members: 

.. autoclass:: transformers.models.led.modeling_tf_led.TFLEDEncoderBaseModelOutput
    :members: 

.. autoclass:: transformers.models.led.modeling_tf_led.TFLEDSeq2SeqModelOutput
    :members: 

.. autoclass:: transformers.models.led.modeling_tf_led.TFLEDSeq2SeqLMOutput
    :members: 




LEDModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LEDModel
    :members: forward


LEDForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LEDForConditionalGeneration
    :members: forward


LEDForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LEDForSequenceClassification
    :members: forward


LEDForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LEDForQuestionAnswering
    :members: forward


TFLEDModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLEDModel
    :members: call


TFLEDForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLEDForConditionalGeneration
    :members: call
