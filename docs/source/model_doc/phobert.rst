PhoBERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The PhoBERT model is presented in the paper `PhoBERT: Pre-trained language models for Vietnamese <https://arxiv.org/abs/2003.00744>`_ by Dat Quoc Nguyen and Anh Tuan Nguyen.
PhoBERT pre-training procedure is based on RoBERTa which optimizes the BERT pre-training approach for more robust performance.

The abstract from the paper is the following:

*We present PhoBERT with two versions, PhoBERT-base and PhoBERT-large, the first public large-scale monolingual language models pre-trained for Vietnamese. Experimental results show that PhoBERT consistently outperforms the recent best pre-trained multilingual model XLM-R (Conneau et al., 2020) and improves the state-of-the-art in multiple Vietnamese-specific NLP tasks including Part-of-speech tagging, Dependency parsing, Named-entity recognition and Natural language inference. We release PhoBERT to facilitate future research and downstream applications for Vietnamese NLP. Our PhoBERT models are available at https://github.com/VinAIResearch/PhoBERT.*

The original code can be found `here <https://github.com/VinAIResearch/PhoBERT>`_.

PhobertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertConfig
    :members:


PhobertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


PhobertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertModel
    :members:


PhobertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertForMaskedLM
    :members:


PhobertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertForSequenceClassification
    :members:


PhobertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertForMultipleChoice
    :members:


PhobertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertForTokenClassification
    :members:


PhobertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PhobertForQuestionAnswering
    :members:


TFPhobertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPhobertModel
    :members:


TFPhobertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPhobertForMaskedLM
    :members:


TFPhobertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPhobertForSequenceClassification
    :members:


TFPhobertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPhobertForMultipleChoice
    :members:


TFPhobertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPhobertForTokenClassification
    :members:


TFPhobertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPhobertForQuestionAnswering
    :members: