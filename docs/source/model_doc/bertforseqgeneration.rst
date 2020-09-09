BERT
----------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~

The BertForSeqGeneration model is a BERT model that can be leveraged for sequence-to-sequence tasks using :class:`~transformers.EncoderDecoderModel` as proposed in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

The abstract from the paper is the following:

*Unsupervised pre-training of large neural models has recently revolutionized Natural Language Processing. By warm-starting from the publicly released checkpoints, NLP practitioners have pushed the state-of-the-art on multiple benchmarks while saving significant amounts of compute time. So far the focus has been mainly on the Natural Language Understanding tasks. In this paper, we demonstrate the efficacy of pre-trained checkpoints for Sequence Generation. We developed a Transformer-based sequence-to-sequence model that is compatible with publicly available pre-trained BERT, GPT-2 and RoBERTa checkpoints and conducted an extensive empirical study on the utility of initializing our model, both encoder and decoder, with these checkpoints. Our models result in new state-of-the-art results on Machine Translation, Text Summarization, Sentence Splitting, and Sentence Fusion.*

Tips:

- :class:`~transformers.BertForSeqGenerationEncoderModel` and :class:`~transformers.BertForSeqGenerationDecoder`  should be used in combination with :class:`~transformers.EncoderDecoder`.
  the right rather than the left.
- For summarization, sentence splitting, sentence fusion and translation, no special tokens are required for the input. Therefore, no EOS token should be added to the end of the input.

The original code can be found `here <https://tfhub.dev/s?module-type=text-generation&subtype=module,placeholder>`_.

BertForSeqGenerationConfig
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForSeqGenerationConfig
    :members:


BertForSeqGenerationTokenizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForSeqGenerationTokenizer
    :members: 

BertForSeqGenerationEncoderModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForSeqGenerationEncoderModel
    :members:


BertForSeqGenerationDecoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertForSeqGenerationDecoder
    :members:
