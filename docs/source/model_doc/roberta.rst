RoBERTa
----------------------------------------------------

The RoBERTa model was proposed in `RoBERTa: A Robustly Optimized BERT Pretraining Approach`_
by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
Veselin Stoyanov. It is based on Google's BERT model released in 2018.

It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
objective and training with much larger mini-batches and learning rates.

This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained
models.

RobertaConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaConfig
    :members:


RobertaTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaTokenizer
    :members:


RobertaModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaModel
    :members:


RobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForMaskedLM
    :members:


RobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForSequenceClassification
    :members:


RobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForTokenClassification
    :members:

TFRobertaModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaModel
    :members:


TFRobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaForMaskedLM
    :members:


TFRobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaForSequenceClassification
    :members:


TFRobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaForTokenClassification
    :members:
