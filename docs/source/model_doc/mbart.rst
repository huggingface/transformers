MBart
-----------------------------------------------------------------------------------------------------------------------

**DISCLAIMER:** If you see something strange, file a `Github Issue
<https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@patrickvonplaten

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MBart model was presented in `Multilingual Denoising Pre-training for Neural Machine Translation
<https://arxiv.org/abs/2001.08210>`_ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov Marjan
Ghazvininejad, Mike Lewis, Luke Zettlemoyer.

According to the abstract, MBART is a sequence-to-sequence denoising auto-encoder pretrained on large-scale monolingual
corpora in many languages using the BART objective. mBART is one of the first methods for pretraining a complete
sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only
on the encoder, decoder, or reconstructing parts of the text.

The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/mbart>`__

Examples
_______________________________________________________________________________________________________________________

- Examples and scripts for fine-tuning mBART and other models for sequence to sequence tasks can be found in
  `examples/seq2seq/ <https://github.com/huggingface/transformers/blob/master/examples/seq2seq/README.md>`__.
- Given the large embeddings table, mBART consumes a large amount of GPU RAM, especially for fine-tuning.
  :class:`MarianMTModel` is usually a better choice for bilingual machine translation.

Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MBart is a multilingual encoder-decoder (seq-to-seq) model primarily intended for translation task. As the model is
multilingual it expects the sequences in a different format. A special language id token is added in both the source
and target text. The source text format is :obj:`X [eos, src_lang_code]` where :obj:`X` is the source text. The target
text format is :obj:`[tgt_lang_code] X [eos]`. :obj:`bos` is never used.

The :meth:`~transformers.MBartTokenizer.prepare_seq2seq_batch` handles this automatically and should be used to encode
the sequences for sequence-to-sequence fine-tuning.

- Supervised training

.. code-block::

    example_english_phrase = "UN Chief Says There Is No Military Solution in Syria"
    expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
    batch = tokenizer.prepare_seq2seq_batch(example_english_phrase, src_lang="en_XX", tgt_lang="ro_RO", tgt_texts=expected_translation_romanian, return_tensors="pt")
    model(input_ids=batch['input_ids'], labels=batch['labels']) # forward pass

- Generation

    While generating the target text set the :obj:`decoder_start_token_id` to the target language id. The following
    example shows how to translate English to Romanian using the `facebook/mbart-large-en-ro` model.

.. code-block::

    from transformers import MBartForConditionalGeneration, MBartTokenizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
    article = "UN Chief Says There Is No Military Solution in Syria"
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], src_lang="en_XX", return_tensors="pt")
    translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    assert translation == "Şeful ONU declară că nu există o soluţie militară în Siria"


MBartConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MBartConfig
    :members:


MBartTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MBartTokenizer
    :members: build_inputs_with_special_tokens, prepare_seq2seq_batch


MBartForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.MBartForConditionalGeneration
    :members:


TFMBartForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFMBartForConditionalGeneration
    :members:
