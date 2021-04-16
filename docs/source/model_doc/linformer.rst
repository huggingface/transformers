.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Linformer
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Linformer model was proposed in `Linformer: Self-Attention with Linear Complexity
<https://arxiv.org/abs/2006.04768>`__  by Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma. Transformers are notoriously 
resource-intensive because their self-attention mechanism requires a squared number of memory and computations in the length of 
the input sequence. The Linformer model gets around that by using the fact that often, the actual information in the attention 
matrix is of lower rank and can be approximated. This is done by projecting the keys and values to a lower dimension along the 
sequence dimension. 

The abstract from the paper is the following:

*Large transformer models have shown extraordinary success in achieving state-of-the-art results in many natural language processing 
applications. However, training and deploying these models can be prohibitively costly for long sequences, as the standard self-attention 
mechanism of the Transformer uses O(n2) time and space with respect to sequence length. In this paper, we demonstrate that the 
self-attention mechanism can be approximated by a low-rank matrix. We further exploit this finding to propose a new self-attention 
mechanism, which reduces the overall self-attention complexity from O(n2) to O(n) in both time and space. The resulting linear 
transformer, the Linformer, performs on par with standard Transformer models, while being much more memory- and time-efficient.*

Tips:

- Linformer is based on BERT, the only difference being the self-attention mechanism. In Linformer, this self-attention mechanism 
  scales linear with the sequence length rather than quadratic. This allows us to train this model with larger sequence lengths. 
- Linformer does not work for the auto-regressive case (i.e. it cannot be used as a decoder), since this model projects along the 
  sequence dimension.  
- Linformer assumes a fixed sequence length for its inputs, which should be specified as the ``seq_len`` attribute of :class:`~transformers.LinformerConfig`. 
  It is set to ``512`` by default.
- Linformer projects the keys and values of the self-attention layers to a lower dimension along the sequence length dimension. This lower
  dimension can be specified as the ``k`` attribute of :class:`~transformers.LinformerConfig`, which is set to ``256`` by default.

The original code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/linformer>`__.

LinformerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerConfig
    :members:


LinformerTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


LinformerTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerTokenizerFast
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


LinformerModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerModel
    :members: forward


LinformerForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerForMaskedLM
    :members: forward


LinformerForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerForSequenceClassification
    :members: forward


LinformerForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerForMultipleChoice
    :members: forward


LinformerForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerForTokenClassification
    :members: forward


LinformerForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LinformerForQuestionAnswering
    :members: forward
    

TFLinformerModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLinformerModel
    :members: call


TFLinformerForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLinformerForMaskedLM
    :members: call


TFLinformerForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLinformerForSequenceClassification
    :members: call


TFLinformerForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLinformerForMultipleChoice
    :members: call


TFLinformerForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLinformerForTokenClassification
    :members: call


TFLinformerForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFLinformerForQuestionAnswering
    :members: call