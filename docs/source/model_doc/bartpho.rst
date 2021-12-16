..
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

BARTpho
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The BARTpho model was proposed in `BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese
<https://arxiv.org/abs/2109.09701>`__ by Nguyen Luong Tran, Duong Minh Le and Dat Quoc Nguyen.

The abstract from the paper is the following:

*We present BARTpho with two versions -- BARTpho_word and BARTpho_syllable -- the first public large-scale monolingual
sequence-to-sequence models pre-trained for Vietnamese. Our BARTpho uses the "large" architecture and pre-training
scheme of the sequence-to-sequence denoising model BART, thus especially suitable for generative NLP tasks. Experiments
on a downstream task of Vietnamese text summarization show that in both automatic and human evaluations, our BARTpho
outperforms the strong baseline mBART and improves the state-of-the-art. We release BARTpho to facilitate future
research and applications of generative Vietnamese NLP tasks.*

Example of use:

.. code-block::

    >>> import torch
    >>> from transformers import AutoModel, AutoTokenizer

    >>> bartpho = AutoModel.from_pretrained("vinai/bartpho-syllable")

    >>> tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    >>> line = "Chúng tôi là những nghiên cứu viên."

    >>> input_ids = tokenizer(line, return_tensors="pt")

    >>> with torch.no_grad():
    ...     features = bartpho(**input_ids)  # Models outputs are now tuples

    >>> # With TensorFlow 2.0+:
    >>> from transformers import TFAutoModel
    >>> bartpho = TFAutoModel.from_pretrained("vinai/bartpho-syllable")
    >>> input_ids = tokenizer(line, return_tensors="tf")
    >>> features = bartpho(**input_ids)

Tips:

- Following mBART, BARTpho uses the "large" architecture of BART with an additional layer-normalization layer on top of
  both the encoder and decoder. Thus, usage examples in the :doc:`documentation of BART <bart>`, when adapting to use
  with BARTpho, should be adjusted by replacing the BART-specialized classes with the mBART-specialized counterparts.
  For example:

.. code-block::

    >>> from transformers import MBartForConditionalGeneration
    >>> bartpho = MBartForConditionalGeneration.from_pretrained("vinai/bartpho-syllable")
    >>> TXT = 'Chúng tôi là <mask> nghiên cứu viên.'
    >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
    >>> logits = bartpho(input_ids).logits
    >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    >>> probs = logits[0, masked_index].softmax(dim=0)
    >>> values, predictions = probs.topk(5)
    >>> print(tokenizer.decode(predictions).split())

- This implementation is only for tokenization: "monolingual_vocab_file" consists of Vietnamese-specialized types
  extracted from the pre-trained SentencePiece model "vocab_file" that is available from the multilingual XLM-RoBERTa.
  Other languages, if employing this pre-trained multilingual SentencePiece model "vocab_file" for subword
  segmentation, can reuse BartphoTokenizer with their own language-specialized "monolingual_vocab_file".

This model was contributed by `dqnguyen <https://huggingface.co/dqnguyen>`__. The original code can be found `here
<https://github.com/VinAIResearch/BARTpho>`__.

BartphoTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartphoTokenizer
    :members:
