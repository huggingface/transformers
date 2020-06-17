RetriBERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The RetriBERT model was proposed in the blog post
`Explain Anything Like I'm Five: A Model for Open Domain Long Form Question Answering <https://yjernite.github.io/lfqa.html>`__,
RetriBERT is a small model that uses either a single or pair of Bert encoders with lower-dimension projection for dense semantic indexing of text.

Code to train and use the model can be found `here <https://github.com/huggingface/transformers/tree/master/examples/distillation>`_.


RetriBertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RetriBertConfig
    :members:


RetriBertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RetriBertTokenizer
    :members:


RetriBertTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RetriBertTokenizerFast
    :members:


RetriBertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RetriBertModel
    :members:
