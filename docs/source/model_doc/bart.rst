Bart
----------------------------------------------------
**DISCLAIMER:** This model is still a work in progress, if you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@sshleifer

Paper
~~~~~
The Bart model was `proposed <https://arxiv.org/abs/1910.13461>`_ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.
According to the abstract:

- Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT).
- The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.
- BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/bart>`_


Implementation Notes
~~~~~~~~~~~~~~~~~~~~
- Bart doesn't use :obj:`token_type_ids`, for sequence classification just use BartTokenizer.encode to get the proper splitting.
- Inputs to the decoder are created by BartModel.forward if they are not passed. This is different than some other model APIs.
- Model predictions are intended to be identical to the original implementation. This only works, however, if the string you pass to fairseq.encode starts with a space.
- Decoder inputs are created automatically by the helper function ``transformers.modeling_bart._prepare_bart_decoder_inputs``
BartModel
- ``MaskedLM.generate`` should be used for summarization, see the example in that docstrings


BartModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartModel
    :members: forward


BartForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForMaskedLM
    :members: forward, generate


BartForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForSequenceClassification
    :members: forward

BartConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartConfig
    :members:

Automatic Creation of Decoder Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is enabled by default

.. autofunction:: transformers.modeling_bart._prepare_bart_decoder_inputs
