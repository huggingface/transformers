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

    >>> # tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    >>> line = "Chúng tôi là những nghiên cứu viên."

    >>> input_ids = torch.tensor([tokenizer.encode(line)])

    >>> with torch.no_grad():
    ...     features = bartpho(input_ids)  # Models outputs are now tuples

    >>> # With TensorFlow 2.0+:
    >>> # from transformers import TFAutoModel
    >>> # bartpho = TFAutoModel.from_pretrained("vinai/bartpho-syllable")

This model was contributed by `dqnguyen <https://huggingface.co/dqnguyen>`__. The original code can be found `here
<https://github.com/VinAIResearch/BARTpho>`__.

BartphoTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BartphoTokenizer
    :members:
