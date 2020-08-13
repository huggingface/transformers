MBart
----------------------------------------------------
**DISCLAIMER:** If you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@sshleifer

Overview
~~~~~~~~~~~~~~~~~~~~~
The MBart model was presented in `Multilingual Denoising Pre-training for Neural Machine Translation <https://arxiv.org/abs/2001.08210>`_ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov
Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer. According to the abstract,

MBART is a sequence-to-sequence denoising auto-encoder pre-trained on large-scale monolingual corpora in many languages using the BART objective. mBART is one of the first methods for pre-training a complete sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only on the encoder, decoder, or reconstructing parts of the text.

The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/mbart>`__


MBartConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MBartConfig
    :members:


MBartTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MBartTokenizer
    :members: build_inputs_with_special_tokens, prepare_seq2seq_batch


MBartForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MBartForConditionalGeneration
    :members: generate, forward


