XLM
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The XLM model was proposed in `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`_
by Guillaume Lample*, Alexis Conneau*. It's a transformer pre-trained using one of the following objectives:

- a causal language modeling (CLM) objective (next token prediction),
- a masked language modeling (MLM) objective (Bert-like), or
- a Translation Language Modeling (TLM) object (extension of Bert's MLM to multiple language inputs)

The abstract from the paper is the following:

*Recent studies have demonstrated the efficiency of generative pretraining for English natural language understanding.
In this work, we extend this approach to multiple languages and show the effectiveness of cross-lingual pretraining.
We propose two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual
data, and one supervised that leverages parallel data with a new cross-lingual language model objective. We obtain
state-of-the-art results on cross-lingual classification, unsupervised and supervised machine translation. On XNLI,
our approach pushes the state of the art by an absolute gain of 4.9% accuracy. On unsupervised machine translation,
we obtain 34.3 BLEU on WMT'16 German-English, improving the previous state of the art by more than 9 BLEU. On
supervised machine translation, we obtain a new state of the art of 38.5 BLEU on WMT'16 Romanian-English, outperforming
the previous best approach by more than 4 BLEU. Our code and pretrained models will be made publicly available.*

Tips:

- XLM has many different checkpoints, which were trained using different objectives: CLM, MLM or TLM. Make sure to
  select the correct objective for your task (e.g. MLM checkpoints are not suitable for generation).
- XLM has multilingual checkpoints which leverage a specific `lang` parameter. Check out the
  `multi-lingual <../multilingual.html>`__ page for more information.


XLMConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMConfig
    :members:

XLMTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMTokenizer
    :members:

XLMModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMModel
    :members:


XLMWithLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMWithLMHeadModel
    :members:


XLMForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMForSequenceClassification
    :members:


XLMForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMForQuestionAnsweringSimple
    :members:


XLMForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.XLMForQuestionAnswering
    :members:


TFXLMModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMModel
    :members:


TFXLMWithLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMWithLMHeadModel
    :members:


TFXLMForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMForSequenceClassification
    :members:


TFXLMForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFXLMForQuestionAnsweringSimple
    :members:
