Longformer
----------------------------------------------------
**DISCLAIMER:** This model is still a work in progress, if you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`_

Overview
~~~~~~~~~
The Longformer model was presented in `Longformer: The Long-Document Transformer <https://arxiv.org/pdf/2004.05150.pdf>`_ by Iz Beltagy, Matthew E. Peters, Arman Cohan.
Here the abstract: 

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA.*

The Authors' code can be found `here <https://github.com/allenai/longformer>`_ .

Longformer Self Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~
Longformer self attention employs self attention on both a "local" context and a "global" context.
Most tokens only attend "locally" to each other meaning that each token attends to its :math:`\frac{1}{2} w` previous tokens and :math:`\frac{1}{2} w` succeding tokens with :math:`w` being the window length as defined in `config.attention_window`. Note that `config.attention_window` can be of type ``list`` to define a different :math:`w` for each layer. 
A selecetd few tokens attend "globally" to all other tokens, as it is conventionally done for all tokens in *e.g.* `BertSelfAttention`.

Note that "locally" and "globally" attending tokens are projected by different query, key and value matrices.
Also note that every "locally" attending token not only attends to tokens within its window :math:`w`, but also to all "globally" attending tokens so that global attention is *symmetric*.

The user can define which tokens attend "locally" and which tokens attend "globally" by setting the tensor `global_attention_mask` at run-time appropriately. `Longformer` employs the following logic for `global_attention_mask`: `0` - the token attends "locally", `1` - token attends "globally". For more information please also refer to :func:`~transformers.LongformerModel.forward` method.

Using Longformer self attention, the memory and time complexity of the query-key matmul operation, which usually represents the memory and time bottleneck, can be reduced from :math:`\mathcal{O}(n_s \times n_s)` to :math:`\mathcal{O}(n_s \times w)`, with :math:`n_s` being the sequence length and :math:`w` being the average window size. It is assumed that the number of "globally" attending tokens is insignificant as compared to the number of "locally" attending tokens.

For more information, please refer to the official `paper <https://arxiv.org/pdf/2004.05150.pdf>`_ .


Training
~~~~~~~~~~~~~~~~~~~~
``LongformerForMaskedLM`` is trained the exact same way, ``RobertaForMaskedLM`` is trained and 
should be used as follows:

::

  input_ids = tokenizer.encode('This is a sentence from [MASK] training data', return_tensors='pt')
  mlm_labels = tokenizer.encode('This is a sentence from the training data', return_tensors='pt')

  loss = model(input_ids, labels=input_ids, masked_lm_labels=mlm_labels)[0]


LongformerConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerConfig
    :members:


LongformerTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerTokenizer
    :members: 


LongformerTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerTokenizerFast
    :members: 


LongformerModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerModel
    :members:


LongformerForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForMaskedLM
    :members:


LongformerForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForSequenceClassification
    :members:


LongformerForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForMultipleChoice
    :members:


LongformerForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForTokenClassification
    :members:


LongformerForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LongformerForQuestionAnswering
    :members:
