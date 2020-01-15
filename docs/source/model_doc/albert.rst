ALBERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The ALBERT model was proposed in `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations <https://arxiv.org/abs/1909.11942>`_
by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. It presents
two parameter-reduction techniques to lower memory consumption and increase the trainig speed of BERT:

- Splitting the embedding matrix into two smaller matrices
- Using repeating layers split among groups

The abstract from the paper is the following:

*Increasing model size when pretraining natural language representations often results in improved performance on
downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations,
longer training times, and unexpected model degradation. To address these problems, we present two parameter-reduction
techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows
that our proposed methods lead to models that scale much better compared to the original BERT. We also use a
self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream
tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE,
RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.*

Tips:

- ALBERT is a model with absolute position embeddings so it's usually advised to pad the inputs on
  the right rather than the left.
- ALBERT uses repeating layers which results in a small memory footprint, however the computational cost remains
  similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same
  number of (repeating) layers.

AlbertConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AlbertConfig
    :members:


AlbertTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AlbertTokenizer
    :members:


AlbertModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AlbertModel
    :members:


AlbertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AlbertForMaskedLM
    :members:


AlbertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AlbertForSequenceClassification
    :members:


AlbertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AlbertForQuestionAnswering
    :members:


TFAlbertModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFAlbertModel
    :members:


TFAlbertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFAlbertForMaskedLM
    :members:


TFAlbertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFAlbertForSequenceClassification
    :members:
