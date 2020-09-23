FSMT
----------------------------------------------------
**DISCLAIMER:** If you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@stas00.

Overview
~~~~~~~~~~~~~~~~~~~~~

FSMT (FairSeq MachineTranslation) models were introduced in "Facebook FAIR's WMT19 News Translation Task Submission" <this paper <https://arxiv.org/abs/1907.06616>__ by Nathan Ng, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli, Sergey Edunov.

The abstract of the paper is the following:

    This paper describes Facebook FAIR's submission to the WMT19 shared news translation task. We participate in two language pairs and four language directions, English <-> German and English <-> Russian. Following our submission from last year, our baseline systems are large BPE-based transformer models trained with the Fairseq sequence modeling toolkit which rely on sampled back-translations. This year we experiment with different bitext data filtering schemes, as well as with adding filtered back-translated data. We also ensemble and fine-tune our models on domain-specific data, then decode using noisy channel model reranking. Our submissions are ranked first in all four directions of the human evaluation campaign. On En->De, our system significantly outperforms other systems as well as human translations. This system improves upon our WMT'18 submission by 4.5 BLEU points.

The original code can be found here <https://github.com/pytorch/fairseq/tree/master/examples/wmt19>__.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~

- FSMT uses source and target vocab pair, that aren't combined into one. It doesn't share embed tokens either. Its tokenizer is very similar to `XLMTokenizer` and the main model is derived from `BartModel`.


FSMTForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FSMTForConditionalGeneration
    :members: forward


FSMTConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FSMTConfig
    :members:


FSMTTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FSMTTokenizer
    :members:


FSMTModel
~~~~~~~~~~~~~

.. autoclass:: transformers.FSMTModel
    :members: forward
