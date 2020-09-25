MBart
-----------------------------------------------------------------------------------------------------------------------
**DISCLAIMER:** If you see something strange,
file a `Github Issue <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@sshleifer

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The MBart model was presented in `Multilingual Denoising Pre-training for Neural Machine Translation <https://arxiv.org/abs/2001.08210>`_ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov
Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer. According to the abstract,

MBART is a sequence-to-sequence denoising auto-encoder pre-trained on large-scale monolingual corpora in many languages using the BART objective. mBART is one of the first methods for pre-training a complete sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only on the encoder, decoder, or reconstructing parts of the text.

The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/mbart>`__


Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MBart is a multilingual encoder-decoder (seq-to-seq) model primarily intended for translation task. 
As the model is multilingual it expects the sequences in a different format. A special language id token 
is added in both the source and target text. The source text format is ``X [eos, src_lang_code]`` 
where ``X`` is the source text. The target text format is ```[tgt_lang_code] X [eos]```. ```bos``` is never used.
The ```MBartTokenizer.prepare_seq2seq_batch``` handles this automatically and should be used to encode 
the sequences for seq-2-seq fine-tuning.

- Supervised training

.. code-block::

    example_english_phrase = "UN Chief Says There Is No Military Solution in Syria"
    expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
    batch = tokenizer.prepare_seq2seq_batch(example_english_phrase, src_lang="en_XX", tgt_lang="ro_RO", tgt_texts=expected_translation_romanian)
    input_ids = batch["input_ids"]
    target_ids = batch["decoder_input_ids"]
    decoder_input_ids = target_ids[:, :-1].contiguous()
    labels = target_ids[:, 1:].clone()
    model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels) #forward

- Generation

    While generating the target text set the `decoder_start_token_id` to the target language id. 
    The following example shows how to translate English to Romanian using the ```facebook/mbart-large-en-ro``` model.

.. code-block::

    from transformers import MBartForConditionalGeneration, MBartTokenizer
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro")
    article = "UN Chief Says There Is No Military Solution in Syria"
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], src_lang="en_XX")
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
    :members: generate, forward


