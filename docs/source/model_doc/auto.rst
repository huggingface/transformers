AutoModel, AutoConfig and AutoTokenizer - Standard derived classes
---------------------------------------------------------------

In many case, the architecture you want to use can be guessed from the name or the path of the pretrained model you are supplying to the ``from_pretrained`` method.

Auto classes are here to do this job for you so that you automatically retreive the relevant model given the name/path to the pretrained weights/config/vocabulary.

``AutoConfig``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoConfig
    :members:


``AutoModel``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoModel
    :members:


``AutoTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_transformers.AutoTokenizer
    :members:
