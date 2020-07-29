BERTweet
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The BERTweet model is presented in the paper `BERTweet: A pre-trained language model for English Tweets <https://arxiv.org/abs/2005.10200>`_ by Dat Quoc Nguyen, Thanh Vu and Anh Tuan Nguyen.
BERTweet uses the same architecture configuration as BERT-base, which is trained with a masked language modeling objective. BERTweet pre-training procedure is based on RoBERTa which optimizes the BERT pre-training approach for more robust performance.
The corpus used to pre-train BERTweet consists of 850M English Tweets (16B word tokens ~ 80GB), containing 845M Tweets streamed from 01/2012 to 08/2019 and 5M Tweets related the COVID-19 pandemic.

The abstract from the paper is the following:

*We present BERTweet, the first public large-scale pre-trained language model for English Tweets. Our BERTweet is trained using the RoBERTa pre-training procedure (Liu et al., 2019), with the same model configuration as BERT-base (Devlin et al., 2019). Experiments show that BERTweet outperforms strong baselines RoBERTa-base and XLM-R-base (Conneau et al., 2020), producing better performance results than the previous state-of-the-art models on three Tweet NLP tasks: Part-of-speech tagging, Named-entity recognition and text classification. We release BERTweet under the MIT License to facilitate future research and applications on Tweet data. Our BERTweet is available at https://github.com/VinAIResearch/BERTweet.*

The original code can be found `here <https://github.com/VinAIResearch/BERTweet>`_.

BertweetConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetConfig
    :members:


BertweetTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


BertweetModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetModel
    :members:


BertweetForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetForMaskedLM
    :members:


BertweetForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetForSequenceClassification
    :members:


BertweetForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetForMultipleChoice
    :members:


BertweetForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetForTokenClassification
    :members:


BertweetForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertweetForQuestionAnswering
    :members:


TFBertweetModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertweetModel
    :members:


TFBertweetForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertweetForMaskedLM
    :members:


TFBertweetForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertweetForSequenceClassification
    :members:


TFBertweetForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertweetForMultipleChoice
    :members:


TFBertweetForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertweetForTokenClassification
    :members:


TFBertweetForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertweetForQuestionAnswering
    :members: