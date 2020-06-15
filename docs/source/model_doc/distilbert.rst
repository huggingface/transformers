DistilBERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The DistilBERT model was proposed in the blog post
`Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT <https://medium.com/huggingface/distilbert-8cf3380435b5>`__,
and the paper `DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter <https://arxiv.org/abs/1910.01108>`__.
DistilBERT is a small, fast, cheap and light Transformer model trained by distilling Bert base. It has 40% less
parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of Bert's performances as measured on
the GLUE language understanding benchmark.

The abstract from the paper is the following:

*As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP),
operating these large models in on-the-edge and/or under constrained computational training or inference budgets
remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation
model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger
counterparts. While most prior work investigated the use of distillation for building task-specific models, we
leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a
BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage
the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language
modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train
and we demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative
on-device study.*

Tips:

- DistilBert doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just separate your segments with the separation token `tokenizer.sep_token` (or `[SEP]`)
- DistilBert doesn't have options to select the input positions (`position_ids` input). This could be added if necessary though, just let's us know if you need this option.

The original code can be found `here <https://github.com/huggingface/transformers/tree/master/examples/distillation>`_.


DistilBertConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertConfig
    :members:


DistilBertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertTokenizer
    :members:


DistilBertTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertTokenizerFast
    :members:


DistilBertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertModel
    :members:


DistilBertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForMaskedLM
    :members:


DistilBertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForSequenceClassification
    :members:


DistilBertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForMultipleChoice
    :members:


DistilBertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForTokenClassification
    :members:


DistilBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.DistilBertForQuestionAnswering
    :members:

TFDistilBertModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertModel
    :members:


TFDistilBertForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForMaskedLM
    :members:


TFDistilBertForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForSequenceClassification
    :members:



TFDistilBertForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForMultipleChoice
    :members:



TFDistilBertForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForTokenClassification
    :members:


TFDistilBertForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFDistilBertForQuestionAnswering
    :members:
