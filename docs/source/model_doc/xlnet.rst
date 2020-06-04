XLNet
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The XLNet model was proposed in `XLNet: Generalized Autoregressive Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`_
by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method
to learn bidirectional contexts by maximizing the expected likelihood over all permutations
of the input sequence factorization order.

The abstract from the paper is the following:

*With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves
better performance than pretraining approaches based on autoregressive language modeling. However, relying on
corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a
pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive
pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over
all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive
formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model,
into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by
a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.*

Tips:

- The specific attention pattern can be controlled at training and test time using the `perm_mask` input.
- Due to the difficulty of training a fully auto-regressive model over various factorization order,
  XLNet is pretrained using only a sub-set of the output tokens as target which are selected
  with the `target_mapping` input.
- To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
  `target_mapping` inputs to control the attention span and outputs (see examples in `examples/text-generation/run_generation.py`)
- XLNet is one of the few models that has no sequence length limit.

The original code can be found `here <https://github.com/zihangdai/xlnet/>`_.


XLNetConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetConfig
    :members:


XLNetTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


XLNetModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetModel
    :members:


XLNetLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetLMHeadModel
    :members:


XLNetForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForSequenceClassification
    :members:


XLNetForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForTokenClassification
    :members:


XLNetForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForMultipleChoice
    :members:


XLNetForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForQuestionAnsweringSimple
    :members:


XLNetForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForQuestionAnswering
    :members:


TFXLNetModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetModel
    :members:


TFXLNetLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetLMHeadModel
    :members:


TFXLNetForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetForSequenceClassification
    :members:


TFXLNetForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetForQuestionAnsweringSimple
    :members:
