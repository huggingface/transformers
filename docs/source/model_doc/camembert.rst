CamemBERT
----------------------------------------------------

The CamemBERT model was proposed in `CamemBERT: a Tasty French Language Model <https://arxiv.org/abs/1911.03894>`__
by Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la
Clergerie, Djamé Seddah, and Benoît Sagot. It is based on Facebook's RoBERTa model released in 2019. It is a model
trained on 138GB of French text.

The abstract from the paper is the following:

*Pretrained language models are now ubiquitous in Natural Language Processing. Despite their success,
most available models have either been trained on English data or on the concatenation of data in multiple
languages. This makes practical use of such models --in all languages except English-- very limited. Aiming
to address this issue for French, we release CamemBERT, a French version of the Bi-directional Encoders for
Transformers (BERT). We measure the performance of CamemBERT compared to multilingual models in multiple
downstream tasks, namely part-of-speech tagging, dependency parsing, named-entity recognition, and natural
language inference. CamemBERT improves the state of the art for most of the tasks considered. We release the
pretrained model for CamemBERT hoping to foster research and downstream applications for French NLP.*

Tips:

- This implementation is the same as RoBERTa. Refer to the `documentation of RoBERTa <./roberta.html>`__ for usage
  examples as well as the information relative to the inputs and outputs.

CamembertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CamembertConfig
    :members:


CamembertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CamembertTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


CamembertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CamembertModel
    :members:


CamembertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CamembertForMaskedLM
    :members:


CamembertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CamembertForSequenceClassification
    :members:


CamembertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CamembertForMultipleChoice
    :members:


CamembertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CamembertForTokenClassification
    :members:


TFCamembertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCamembertModel
    :members:


TFCamembertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCamembertForMaskedLM
    :members:


TFCamembertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCamembertForSequenceClassification
    :members:


TFCamembertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFCamembertForTokenClassification
    :members:
