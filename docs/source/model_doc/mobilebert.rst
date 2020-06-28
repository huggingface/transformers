MobileBERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The MobileBERT model was proposed in `MobileBERT: a Compact Task-Agnostic BERT
for Resource-Limited Devices <https://arxiv.org/abs/2004.02984>`__
by Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. It's a bidirectional transformer
based on the BERT model, which is compressed and accelerated using several approaches.

The abstract from the paper is the following:

*Natural Language Processing (NLP) has recently achieved great success by using huge pre-trained models with hundreds
of millions of parameters. However, these models suffer from heavy model sizes and high latency such that they cannot
be deployed to resource-limited mobile devices. In this paper, we propose MobileBERT for compressing and accelerating
the popular BERT model. Like the original BERT, MobileBERT is task-agnostic, that is, it can be generically applied
to various downstream NLP tasks via simple fine-tuning. Basically, MobileBERT is a thin version of BERT_LARGE, while
equipped with bottleneck structures and a carefully designed balance between self-attentions and feed-forward
networks. To train MobileBERT, we first train a specially designed teacher model, an inverted-bottleneck incorporated
BERT_LARGE model. Then, we conduct knowledge transfer from this teacher to MobileBERT. Empirical studies show that
MobileBERT is 4.3x smaller and 5.5x faster than BERT_BASE while achieving competitive results on well-known
benchmarks. On the natural language inference tasks of GLUE, MobileBERT achieves a GLUEscore o 77.7
(0.6 lower than BERT_BASE), and 62 ms latency on a Pixel 4 phone. On the SQuAD v1.1/v2.0 question answering task,
MobileBERT achieves a dev F1 score of 90.0/79.2 (1.5/2.1 higher than BERT_BASE).*

Tips:

- MobileBERT is a model with absolute position embeddings so it's usually advised to pad the inputs on
  the right rather than the left.
- MobileBERT is similar to BERT and therefore relies on the masked language modeling (MLM) objective.
  It is therefore efficient at predicting masked tokens and at NLU in general, but is not optimal for
  text generation. Models trained with a causal language modeling (CLM) objective are better in that regard.

The original code can be found `here <https://github.com/google-research/mobilebert>`_.

MobileBertConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertConfig
    :members:


MobileBertTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


MobileBertTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertTokenizerFast
    :members:


MobileBertModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertModel
    :members:


MobileBertForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertForPreTraining
    :members:


MobileBertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertForMaskedLM
    :members:


MobileBertForNextSentencePrediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertForNextSentencePrediction
    :members:


MobileBertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertForSequenceClassification
    :members:


MobileBertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertForMultipleChoice
    :members:


MobileBertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertForTokenClassification
    :members:


MobileBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MobileBertForQuestionAnswering
    :members:


TFMobileBertModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertModel
    :members:


TFMobileBertForPreTraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertForPreTraining
    :members:


TFMobileBertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertForMaskedLM
    :members:


TFMobileBertForNextSentencePrediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertForNextSentencePrediction
    :members:


TFMobileBertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertForSequenceClassification
    :members:


TFMobileBertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertForMultipleChoice
    :members:


TFMobileBertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertForTokenClassification
    :members:


TFMobileBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMobileBertForQuestionAnswering
    :members:

