RoBERTa
----------------------------------------------------

The RoBERTa model was proposed in `RoBERTa: A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`_
by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
Veselin Stoyanov. It is based on Google's BERT model released in 2018.

It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
objective and training with much larger mini-batches and learning rates.

The abstract from the paper is the following:

*Language model pretraining has led to significant performance gains but careful comparison between different
approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes,
and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication
study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and
training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of
every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These
results highlight the importance of previously overlooked design choices, and raise questions about the source
of recently reported improvements. We release our models and code.*

Tips:

- This implementation is the same as :class:`~transformers.BertModel` with a tiny embeddings tweak as well as a
  setup for Roberta pretrained models.
- RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a
  different pre-training scheme.
- RoBERTa doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just separate your segments with the separation token `tokenizer.sep_token` (or `</s>`)
- `Camembert <./camembert.html>`__ is a wrapper around RoBERTa. Refer to this page for usage examples.

The original code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/roberta>`_.


RobertaConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaConfig
    :members:


RobertaTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


RobertaTokenizerFast
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaTokenizerFast
    :members: build_inputs_with_special_tokens


RobertaModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaModel
    :members:


RobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForMaskedLM
    :members:


RobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForSequenceClassification
    :members:


RobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaForTokenClassification
    :members:

TFRobertaModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaModel
    :members:


TFRobertaForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaForMaskedLM
    :members:


TFRobertaForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaForSequenceClassification
    :members:


TFRobertaForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFRobertaForTokenClassification
    :members:
