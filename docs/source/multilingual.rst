Multi-lingual models
=======================================================================================================================

Most of the models available in this library are mono-lingual models (English, Chinese and German). A few multi-lingual
models are available and have a different mechanisms than mono-lingual models. This page details the usage of these
models.

The two models that currently support multiple languages are BERT and XLM.

XLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XLM has a total of 10 different checkpoints, only one of which is mono-lingual. The 9 remaining model checkpoints can
be split in two categories: the checkpoints that make use of language embeddings, and those that don't

XLM & Language Embeddings
-----------------------------------------------------------------------------------------------------------------------

This section concerns the following checkpoints:

- ``xlm-mlm-ende-1024`` (Masked language modeling, English-German)
- ``xlm-mlm-enfr-1024`` (Masked language modeling, English-French)
- ``xlm-mlm-enro-1024`` (Masked language modeling, English-Romanian)
- ``xlm-mlm-xnli15-1024`` (Masked language modeling, XNLI languages)
- ``xlm-mlm-tlm-xnli15-1024`` (Masked language modeling + Translation, XNLI languages)
- ``xlm-clm-enfr-1024`` (Causal language modeling, English-French)
- ``xlm-clm-ende-1024`` (Causal language modeling, English-German)

These checkpoints require language embeddings that will specify the language used at inference time. These language
embeddings are represented as a tensor that is of the same shape as the input ids passed to the model. The values in
these tensors depend on the language used and are identifiable using the ``lang2id`` and ``id2lang`` attributes from
the tokenizer.

Here is an example using the ``xlm-clm-enfr-1024`` checkpoint (Causal language modeling, English-French):


.. code-block::

    >>> import torch
    >>> from transformers import XLMTokenizer, XLMWithLMHeadModel

    >>> tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
    >>> model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")


The different languages this model/tokenizer handles, as well as the ids of these languages are visible using the
``lang2id`` attribute:

.. code-block::

    >>> print(tokenizer.lang2id)
    {'en': 0, 'fr': 1}


These ids should be used when passing a language parameter during a model pass. Let's define our inputs:

.. code-block::

    >>> input_ids = torch.tensor([tokenizer.encode("Wikipedia was used to")]) # batch size of 1


We should now define the language embedding by using the previously defined language id. We want to create a tensor
filled with the appropriate language ids, of the same size as input_ids. For english, the id is 0:

.. code-block::

    >>> language_id = tokenizer.lang2id['en']  # 0
    >>> langs = torch.tensor([language_id] * input_ids.shape[1])  # torch.tensor([0, 0, 0, ..., 0])

    >>> # We reshape it to be of size (batch_size, sequence_length)
    >>> langs = langs.view(1, -1) # is now of shape [1, sequence_length] (we have a batch size of 1)


You can then feed it all as input to your model:

.. code-block::

    >>> outputs = model(input_ids, langs=langs)


The example `run_generation.py
<https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py>`__ can generate
text using the CLM checkpoints from XLM, using the language embeddings.

XLM without Language Embeddings
-----------------------------------------------------------------------------------------------------------------------

This section concerns the following checkpoints:

- ``xlm-mlm-17-1280`` (Masked language modeling, 17 languages)
- ``xlm-mlm-100-1280`` (Masked language modeling, 100 languages)

These checkpoints do not require language embeddings at inference time. These models are used to have generic sentence
representations, differently from previously-mentioned XLM checkpoints.


BERT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BERT has two checkpoints that can be used for multi-lingual tasks:

- ``bert-base-multilingual-uncased`` (Masked language modeling + Next sentence prediction, 102 languages)
- ``bert-base-multilingual-cased`` (Masked language modeling + Next sentence prediction, 104 languages)

These checkpoints do not require language embeddings at inference time. They should identify the language used in the
context and infer accordingly.

XLM-RoBERTa
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

XLM-RoBERTa was trained on 2.5TB of newly created clean CommonCrawl data in 100 languages. It provides strong gains
over previously released multi-lingual models like mBERT or XLM on downstream tasks like classification, sequence
labeling and question answering.

Two XLM-RoBERTa checkpoints can be used for multi-lingual tasks:

- ``xlm-roberta-base`` (Masked language modeling, 100 languages)
- ``xlm-roberta-large`` (Masked language modeling, 100 languages)
