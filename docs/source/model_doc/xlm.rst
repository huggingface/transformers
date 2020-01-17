XLM
----------------------------------------------------

The XLM model was proposed in `Cross-lingual Language Model Pretraining`_
by Guillaume Lample*, Alexis Conneau*. It's a transformer pre-trained using one of the following objectives:

    - a causal language modeling (CLM) objective (next token prediction),
    - a masked language modeling (MLM) objective (Bert-like), or
    - a Translation Language Modeling (TLM) object (extension of Bert's MLM to multiple language inputs)

Original code can be found `here <https://github.com/facebookresearch/XLM>`_.

This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
refer to the PyTorch documentation for all matter related to general usage and behavior.


XLMConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMConfig
    :members:

XLMTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMTokenizer
    :members:

XLMModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMModel
    :members:


XLMWithLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMWithLMHeadModel
    :members:


XLMForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMForSequenceClassification
    :members:


XLMForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMForQuestionAnsweringSimple
    :members:


XLMForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMForQuestionAnswering
    :members:


TFXLMModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMModel
    :members:


TFXLMWithLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMWithLMHeadModel
    :members:


TFXLMForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMForSequenceClassification
    :members:


TFXLMForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMForQuestionAnsweringSimple
    :members:
