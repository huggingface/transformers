BERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The BERT model was proposed in `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__
by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
pre-trained using a combination of masked language modeling objective and next sentence prediction
on a large corpus comprising the Toronto Book Corpus and Wikipedia.

The abstract from the paper is the following:

*We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations
from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result,
the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models
for a wide range of tasks, such as question answering and language inference, without substantial task-specific
architecture modifications.*

*BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural
language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI
accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute
improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).*

Tips:

- BERT is a model with absolute position embeddings so it's usually advised to pad the inputs on
  the right rather than the left.
- BERT was trained with a masked language modeling (MLM) objective. It is therefore efficient at predicting masked
  tokens and at NLU in general, but is not optimal for text generation. Models trained with a causal language
  modeling (CLM) objective are better in that regard.
- Alongside MLM, BERT was trained using a next sentence prediction (NSP) objective using the [CLS] token as a sequence
  approximate. The user may use this token (the first token in a sequence built with special tokens) to get a sequence
  prediction rather than a token prediction. However, averaging over the sequence may yield better results than using
  the [CLS] token.

The original code can be found `here <https://github.com/google-research/bert>`_.

BertConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertConfig
    :members:


BertTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


BertTokenizerFast
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertTokenizerFast
    :members:


BertModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertModel
    :members:


BertForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForPreTraining
    :members:


BertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForMaskedLM
    :members:


BertForNextSentencePrediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForNextSentencePrediction
    :members:


BertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForSequenceClassification
    :members:


BertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForMultipleChoice
    :members:


BertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForTokenClassification
    :members:


BertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForQuestionAnswering
    :members:


TFBertModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertModel
    :members:


TFBertForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertForPreTraining
    :members:


TFBertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertForMaskedLM
    :members:


TFBertForNextSentencePrediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertForNextSentencePrediction
    :members:


TFBertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertForSequenceClassification
    :members:


TFBertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertForMultipleChoice
    :members:


TFBertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertForTokenClassification
    :members:


TFBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBertForQuestionAnswering
    :members:

