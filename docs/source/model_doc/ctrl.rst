CTRL
----------------------------------------------------

CTRL model was proposed in `CTRL: A Conditional Transformer Language Model for Controllable Generation <https://arxiv.org/abs/1909.05858>`_
by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
It's a causal (unidirectional) transformer pre-trained using language modeling on a very large
corpus of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
usage and behavior.

Note: if you fine-tune a CTRL model using the Salesforce code (https://github.com/salesforce/ctrl),
you'll be able to convert from TF to our HuggingFace/Transformers format using the 
``convert_tf_to_huggingface_pytorch.py`` script (see `issue #1654 <https://github.com/huggingface/transformers/issues/1654>`_).


``CTRLConfig``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLConfig
    :members:


``CTRLTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLTokenizer
    :members:


``CTRLModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLModel
    :members:


``CTRLLMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CTRLLMHeadModel
    :members:


``TFCTRLModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCTRLModel
    :members:


``TFCTRLLMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCTRLLMHeadModel
    :members:

