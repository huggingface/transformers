.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Longformer
-----------------------------------------------------------------------------------------------------------------------

**DISCLAIMER:** This model is still a work in progress, if you see something strange, file a `Github Issue
<https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__.

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Longformer model was presented in `Longformer: The Long-Document Transformer
<https://arxiv.org/pdf/2004.05150.pdf>`__ by Iz Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales
quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention
mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or
longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local
windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we
evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In
contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our
pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on
WikiHop and TriviaQA.*

The Authors' code can be found `here <https://github.com/allenai/longformer>`__.

Longformer Self Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Longformer self attention employs self attention on both a "local" context and a "global" context. Most tokens only
attend "locally" to each other meaning that each token attends to its :math:`\frac{1}{2} w` previous tokens and
:math:`\frac{1}{2} w` succeding tokens with :math:`w` being the window length as defined in
:obj:`config.attention_window`. Note that :obj:`config.attention_window` can be of type :obj:`List` to define a
different :math:`w` for each layer. A selected few tokens attend "globally" to all other tokens, as it is
conventionally done for all tokens in :obj:`BertSelfAttention`.

Note that "locally" and "globally" attending tokens are projected by different query, key and value matrices. Also note
that every "locally" attending token not only attends to tokens within its window :math:`w`, but also to all "globally"
attending tokens so that global attention is *symmetric*.

The user can define which tokens attend "locally" and which tokens attend "globally" by setting the tensor
:obj:`global_attention_mask` at run-time appropriately. All Longformer models employ the following logic for
:obj:`global_attention_mask`:

- 0: the token attends "locally",
- 1: the token attends "globally".

For more information please also refer to :meth:`~transformers.LongformerModel.forward` method.

Using Longformer self attention, the memory and time complexity of the query-key matmul operation, which usually
represents the memory and time bottleneck, can be reduced from :math:`\mathcal{O}(n_s \times n_s)` to
:math:`\mathcal{O}(n_s \times w)`, with :math:`n_s` being the sequence length and :math:`w` being the average window
size. It is assumed that the number of "globally" attending tokens is insignificant as compared to the number of
"locally" attending tokens.

For more information, please refer to the official `paper <https://arxiv.org/pdf/2004.05150.pdf>`__.


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~transformers.LongformerForMaskedLM` is trained the exact same way :class:`~transformers.RobertaForMaskedLM` is
trained and should be used as follows:

.. code-block::

    input_ids = tokenizer.encode('This is a sentence from [MASK] training data', return_tensors='pt')
    mlm_labels = tokenizer.encode('This is a sentence from the training data', return_tensors='pt')

    loss = model(input_ids, labels=input_ids, masked_lm_labels=mlm_labels)[0]


LongformerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerConfig
    :members:


LongformerTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerTokenizer
    :members: 


LongformerTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerTokenizerFast
    :members: 

Longformer specific outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.models.longformer.modeling_longformer.LongformerBaseModelOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling
    :members: 

.. autoclass:: transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_longformer.LongformerSequenceClassifierOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_longformer.LongformerTokenClassifierOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutputWithPooling
    :members: 

.. autoclass:: transformers.models.longformer.modeling_tf_longformer.TFLongformerMaskedLMOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_tf_longformer.TFLongformerQuestionAnsweringModelOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_tf_longformer.TFLongformerSequenceClassifierOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_tf_longformer.TFLongformerMultipleChoiceModelOutput
    :members: 

.. autoclass:: transformers.models.longformer.modeling_tf_longformer.TFLongformerTokenClassifierOutput
    :members: 

LongformerModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerModel
    :members: forward


LongformerForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForMaskedLM
    :members: forward


LongformerForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForSequenceClassification
    :members: forward


LongformerForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForMultipleChoice
    :members: forward


LongformerForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForTokenClassification
    :members: forward


LongformerForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForQuestionAnswering
    :members: forward


TFLongformerModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLongformerModel
    :members: call


TFLongformerForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLongformerForMaskedLM
    :members: call


TFLongformerForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLongformerForQuestionAnswering
    :members: call


TFLongformerForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLongformerForSequenceClassification
    :members: call


TFLongformerForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLongformerForTokenClassification
    :members: call


TFLongformerForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLongformerForMultipleChoice
    :members: call

