Bart
----------------------------------------------------
**DISCLAIMER:** This model is still a work in progress, if you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@sshleifer

Paper
~~~~~
The Bart model was `proposed <https://arxiv.org/abs/1910.13461>`_ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.
According to the abstract,

- Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT).
- The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.
- BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/bart>`_


Implementation Notes
~~~~~~~~~~~~~~~~~~~~
- Bart doesn't use :obj:`token_type_ids` for sequence classification. Use BartTokenizer.encode to get the proper splitting.
- The forward pass of ``BartModel`` will create decoder inputs (using the helper function ``transformers.modeling_bart._prepare_bart_decoder_inputs``)  if they are not passed. This is different than some other modeling APIs.
- Model predictions are intended to be identical to the original implementation. This only works, however, if the string you pass to ``fairseq.encode`` starts with a space.
- ``BartForConditionalGeneration.generate`` should be used for conditional generation tasks like summarization, see the example in that docstrings
- Models that load the ``"bart-large-cnn"`` weights will not have a ``mask_token_id``, or be able to perform mask filling tasks.



BartModel
~~~~~~~~~~~~~

.. autoclass:: transformers.BartModel
    :members: forward

.. autofunction:: transformers.modeling_bart._prepare_bart_decoder_inputs


BartForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForConditionalGeneration
    :members: generate, forward


BartForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForSequenceClassification
    :members: forward

BartConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartConfig
    :members:

