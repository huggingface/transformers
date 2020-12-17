.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

BART
-----------------------------------------------------------------------------------------------------------------------

**DISCLAIMER:** If you see something strange, file a `Github Issue
<https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__ and assign
@patrickvonplaten

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Bart model was proposed in `BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
Translation, and Comprehension <https://arxiv.org/abs/1910.13461>`__ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.

According to the abstract,

- Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a
  left-to-right decoder (like GPT).
- The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme,
  where spans of text are replaced with a single mask token.
- BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It
  matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new
  state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains
  of up to 6 ROUGE.

The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/bart>`__.


Examples
_______________________________________________________________________________________________________________________

- Examples and scripts for fine-tuning BART and other models for sequence to sequence tasks can be found in
  `examples/seq2seq/ <https://github.com/huggingface/transformers/blob/master/examples/seq2seq/README.md>`__.
- An example of how to train :class:`~transformers.BartForConditionalGeneration` with a Hugging Face :obj:`datasets`
  object can be found in this `forum discussion
  <https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904>`__.
- `Distilled checkpoints <https://huggingface.co/models?search=distilbart>`__ are described in this `paper
  <https://arxiv.org/abs/2010.13002>`__.


Implementation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Bart doesn't use :obj:`token_type_ids` for sequence classification. Use :class:`~transformers.BartTokenizer` or
  :meth:`~transformers.BartTokenizer.encode` to get the proper splitting.
- The forward pass of :class:`~transformers.BartModel` will create decoder inputs (using the helper function
  :func:`transformers.models.bart.modeling_bart._prepare_bart_decoder_inputs`) if they are not passed. This is
  different than some other modeling APIs.
- Model predictions are intended to be identical to the original implementation when
  :obj:`force_bos_token_to_be_generated=True`. This only works, however, if the string you pass to
  :func:`fairseq.encode` starts with a space.
- :meth:`~transformers.BartForConditionalGeneration.generate` should be used for conditional generation tasks like
  summarization, see the example in that docstrings.
- Models that load the `facebook/bart-large-cnn` weights will not have a :obj:`mask_token_id`, or be able to perform
  mask-filling tasks.
- For training/forward passes that don't involve beam search, pass :obj:`use_cache=False`.

Mask Filling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :obj:`facebook/bart-base` and :obj:`facebook/bart-large` checkpoints can be used to fill multi-token masks.

.. code-block::

    from transformers import BartForConditionalGeneration, BartTokenizer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", force_bos_token_to_be_generated=True)
    tok = BartTokenizer.from_pretrained("facebook/bart-large")
    example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
    batch = tok(example_english_phrase, return_tensors='pt')
    generated_ids = model.generate(batch['input_ids'])
    assert tok.batch_decode(generated_ids, skip_special_tokens=True) == ['UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria']



BartConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartConfig
    :members:


BartTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartTokenizer
    :members:


BartTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartTokenizerFast
    :members:


BartModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartModel
    :members: forward


BartForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForConditionalGeneration
    :members: forward


BartForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForSequenceClassification
    :members: forward


BartForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartForQuestionAnswering
    :members: forward



TFBartModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBartModel
    :members: call


TFBartForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFBartForConditionalGeneration
    :members: call
