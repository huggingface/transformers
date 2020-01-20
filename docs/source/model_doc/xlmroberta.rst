XLM-RoBERTa
------------------------------------------

The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual Representation Learning at Scale`_
by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's RoBERTa model released in 2019.

It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

This implementation is the same as RoBERTa.

This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
refer to the PyTorch documentation for all matter related to general usage and behavior.

.. _`Unsupervised Cross-lingual Representation Learning at Scale`:
    https://arxiv.org/abs/1911.02116

.. _`torch.nn.Module`:
    https://pytorch.org/docs/stable/nn.html#module


XLMRobertaConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaConfig
    :members:


XLMRobertaTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaTokenizer
    :members:


XLMRobertaModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaModel
    :members:


XLMRobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForMaskedLM
    :members:


XLMRobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForSequenceClassification
    :members:


XLMRobertaForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForMultipleChoice
    :members:


XLMRobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMRobertaForTokenClassification
    :members:

