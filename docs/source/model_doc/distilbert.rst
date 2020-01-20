DistilBERT
----------------------------------------------------

DistilBERT is a small, fast, cheap and light Transformer model
trained by distilling Bert base. It has 40% less parameters than
`bert-base-uncased`, runs 60% faster while preserving over 95% of
Bert's performances as measured on the GLUE language understanding benchmark.

Here are the differences between the interface of Bert and DistilBert:

- DistilBert doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just separate your segments with the separation token `tokenizer.sep_token` (or `[SEP]`)
- DistilBert doesn't have options to select the input positions (`position_ids` input). This could be added if necessary though, just let's us know if you need this option.

For more information on DistilBERT, please refer to our
`detailed blog post`_

.. _`detailed blog post`:
    https://medium.com/huggingface/distilbert-8cf3380435b5


``DistilBertConfig``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertConfig
    :members:


``DistilBertTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertTokenizer
    :members:


``DistilBertModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertModel
    :members:


``DistilBertForMaskedLM``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForMaskedLM
    :members:


``DistilBertForSequenceClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForSequenceClassification
    :members:


``DistilBertForQuestionAnswering``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForQuestionAnswering
    :members:

``TFDistilBertModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertModel
    :members:


``TFDistilBertForMaskedLM``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForMaskedLM
    :members:


``TFDistilBertForSequenceClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForSequenceClassification
    :members:


``TFDistilBertForQuestionAnswering``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForQuestionAnswering
    :members:
