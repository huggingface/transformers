BertGeneration
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BertGeneration model is a BERT model that can be leveraged for sequence-to-sequence tasks using
:class:`~transformers.EncoderDecoderModel` as proposed in `Leveraging Pre-trained Checkpoints for Sequence Generation
Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

The abstract from the paper is the following:

*Unsupervised pretraining of large neural models has recently revolutionized Natural Language Processing. By
warm-starting from the publicly released checkpoints, NLP practitioners have pushed the state-of-the-art on multiple
benchmarks while saving significant amounts of compute time. So far the focus has been mainly on the Natural Language
Understanding tasks. In this paper, we demonstrate the efficacy of pre-trained checkpoints for Sequence Generation. We
developed a Transformer-based sequence-to-sequence model that is compatible with publicly available pre-trained BERT,
GPT-2 and RoBERTa checkpoints and conducted an extensive empirical study on the utility of initializing our model, both
encoder and decoder, with these checkpoints. Our models result in new state-of-the-art results on Machine Translation,
Text Summarization, Sentence Splitting, and Sentence Fusion.*

Usage:

- The model can be used in combination with the :class:`~transformers.EncoderDecoderModel` to leverage two pretrained
  BERT checkpoints for subsequent fine-tuning.

.. code-block::

  # leverage checkpoints for Bert2Bert model...
  # use BERT's cls token as BOS token and sep token as EOS token
  encoder = BertGenerationEncoder.from_pretrained("bert-large-uncased", bos_token_id=101, eos_token_id=102)
  # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
  decoder = BertGenerationDecoder.from_pretrained("bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
  bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

  # create tokenizer...
  tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

  input_ids = tokenizer('This is a long article to summarize', add_special_tokens=False, return_tensors="pt").input_ids
  labels = tokenizer('This is a short summary', return_tensors="pt").input_ids

  # train...
  loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
  loss.backward()


- Pretrained :class:`~transformers.EncoderDecoderModel` are also directly available in the model hub, e.g.,


.. code-block::

  # instantiate sentence fusion model
  sentence_fuser = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
  tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

  input_ids = tokenizer('This is the first sentence. This is the second sentence.', add_special_tokens=False, return_tensors="pt").input_ids

  outputs = sentence_fuser.generate(input_ids)

  print(tokenizer.decode(outputs[0]))


Tips:

- :class:`~transformers.BertGenerationEncoder` and :class:`~transformers.BertGenerationDecoder` should be used in
  combination with :class:`~transformers.EncoderDecoder`.
- For summarization, sentence splitting, sentence fusion and translation, no special tokens are required for the input.
  Therefore, no EOS token should be added to the end of the input.

The original code can be found `here <https://tfhub.dev/s?module-type=text-generation&subtype=module,placeholder>`__.

BertGenerationConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertGenerationConfig
    :members:


BertGenerationTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertGenerationTokenizer
    :members: save_vocabulary

BertGenerationEncoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertGenerationEncoder
    :members: forward


BertGenerationDecoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BertGenerationDecoder
    :members: forward
