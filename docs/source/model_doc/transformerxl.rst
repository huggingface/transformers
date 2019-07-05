Transformer XL
----------------------------------------------------


``TransfoXLTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``TransfoXLTokenizer`` perform word tokenization. This tokenizer can be used for adaptive softmax and has utilities for counting tokens in a corpus to create a vocabulary ordered by toekn frequency (for adaptive softmax). See the adaptive softmax paper (\ `Efficient softmax approximation for GPUs <http://arxiv.org/abs/1609.04309>`_\ ) for more details.

The API is similar to the API of ``BertTokenizer`` (see above).

Please refer to the doc strings and code in `\ ``tokenization_transfo_xl.py`` <./pytorch_pretrained_bert/tokenization_transfo_xl.py>`_ for the details of these additional methods in ``TransfoXLTokenizer``.


12. ``TransfoXLModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.TransfoXLModel
    :members:


13. ``TransfoXLLMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.TransfoXLLMHeadModel
    :members:
