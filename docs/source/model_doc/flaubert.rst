FlauBERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The FlauBERT model was proposed in the paper
`FlauBERT: Unsupervised Language Model Pre-training for French <https://arxiv.org/abs/1912.05372>`__ by Hang Le et al.
It's a transformer pre-trained using a masked language modeling (MLM) objective (BERT-like).

The abstract from the paper is the following:

*Language models have become a key step to achieve state-of-the art results in many different Natural Language
Processing (NLP) tasks. Leveraging the huge amount of unlabeled texts nowadays available, they provide an efficient
way to pre-train continuous word representations that can be fine-tuned for a downstream task, along with their
contextualization at the sentence level. This has been widely demonstrated for English using contextualized
representations (Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018; Radford et al., 2018; Devlin et
al., 2019; Yang et al., 2019b). In this paper, we introduce and share FlauBERT, a model learned on a very large
and heterogeneous French corpus. Models of different sizes are trained using the new CNRS (French National Centre
for Scientific Research) Jean Zay supercomputer. We apply our French language models to diverse NLP tasks (text
classification, paraphrasing, natural language inference, parsing, word sense disambiguation) and show that most
of the time they outperform other pre-training approaches. Different versions of FlauBERT as well as a unified
evaluation protocol for the downstream tasks, called FLUE (French Language Understanding Evaluation), are shared
to the research community for further reproducible experiments in French NLP.*

The original code can be found `here <https://github.com/getalp/Flaubert>`_.


FlaubertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaubertConfig
    :members:


FlaubertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaubertTokenizer
    :members:


FlaubertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaubertModel
    :members:


FlaubertWithLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaubertWithLMHeadModel
    :members:


FlaubertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaubertForSequenceClassification
    :members:


FlaubertForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaubertForQuestionAnsweringSimple
    :members:


FlaubertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaubertForQuestionAnswering
    :members:


TFFlaubertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFFlaubertModel
    :members:


TFFlaubertWithLMHeadModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFFlaubertWithLMHeadModel
    :members:


TFFlaubertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFFlaubertForSequenceClassification
    :members:


TFFlaubertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFFlaubertForMultipleChoice
    :members:


TFFlaubertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFFlaubertForTokenClassification
    :members:


TFFlaubertForQuestionAnsweringSimple
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFFlaubertForQuestionAnsweringSimple
    :members: