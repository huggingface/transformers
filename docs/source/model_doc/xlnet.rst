XLNet
----------------------------------------------------

The XLNet model was proposed in `XLNet: Generalized Autoregressive Pretraining for Language Understanding`_
by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method
to learn bidirectional contexts by maximizing the expected likelihood over all permutations
of the input sequence factorization order.

The specific attention pattern can be controlled at training and test time using the `perm_mask` input.

Due to the difficulty of training a fully auto-regressive model over various factorization order,
XLNet is pretrained using only a sub-set of the output tokens as target which are selected
with the `target_mapping` input.

To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
`target_mapping` inputs to control the attention span and outputs (see examples in `examples/run_generation.py`)


``XLNetConfig``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetConfig
    :members:


``XLNetTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetTokenizer
    :members:


``XLNetModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetModel
    :members:


``XLNetLMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetLMHeadModel
    :members:


``XLNetForSequenceClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForSequenceClassification
    :members:


``XLNetForTokenClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForTokenClassification
    :members:


``XLNetForMultipleChoice``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForMultipleChoice
    :members:


``XLNetForQuestionAnsweringSimple``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForQuestionAnsweringSimple
    :members:


``XLNetForQuestionAnswering``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLNetForQuestionAnswering
    :members:


``TFXLNetModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetModel
    :members:


``TFXLNetLMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetLMHeadModel
    :members:


``TFXLNetForSequenceClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetForSequenceClassification
    :members:


``TFXLNetForQuestionAnsweringSimple``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLNetForQuestionAnsweringSimple
    :members:
