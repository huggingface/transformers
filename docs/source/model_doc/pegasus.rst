.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Pegasus
-----------------------------------------------------------------------------------------------------------------------

**DISCLAIMER:** If you see something strange, file a `Github Issue
<https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title>`__
and assign @patrickvonplaten.


Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Pegasus model was proposed in `PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization
<https://arxiv.org/pdf/1912.08777.pdf>`__ by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019.

According to the abstract,

- Pegasus' pretraining task is intentionally similar to summarization: important sentences are removed/masked from an
  input document and are generated together as one output sequence from the remaining sentences, similar to an
  extractive summary.
- Pegasus achieves SOTA summarization performance on all 12 downstream tasks, as measured by ROUGE and human eval.

This model was contributed by `sshleifer <https://huggingface.co/sshleifer>`__. The Authors' code can be found `here
<https://github.com/google-research/pegasus>`__.


Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the `checkpoints <https://huggingface.co/models?search=pegasus>`__ are fine-tuned for summarization, besides
`pegasus-large`, whence the other checkpoints are fine-tuned:

- Each checkpoint is 2.2 GB on disk and 568M parameters.
- FP16 is not supported (help/ideas on this appreciated!).
- Summarizing xsum in fp32 takes about 400ms/sample, with default parameters on a v100 GPU.
- Full replication results and correctly pre-processed data can be found in this `Issue
  <https://github.com/huggingface/transformers/issues/6844#issue-689259666>`__.
- `Distilled checkpoints <https://huggingface.co/models?search=distill-pegasus>`__ are described in this `paper
  <https://arxiv.org/abs/2010.13002>`__.

Examples
_______________________________________________________________________________________________________________________

- :prefix_link:`Script <examples/research_projects/seq2seq-distillation/finetune_pegasus_xsum.sh>` to fine-tune pegasus
  on the XSUM dataset. Data download instructions at :prefix_link:`examples/pytorch/summarization/
  <examples/pytorch/summarization/README.md>`.
- FP16 is not supported (help/ideas on this appreciated!).
- The adafactor optimizer is recommended for pegasus fine-tuning.


Implementation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All models are transformer encoder-decoders with 16 layers in each component.
- The implementation is completely inherited from :class:`~transformers.BartForConditionalGeneration`
- Some key configuration differences:

    - static, sinusoidal position embeddings
    - the model starts generating with pad_token_id (which has 0 token_embedding) as the prefix.
    - more beams are used (:obj:`num_beams=8`)
- All pretrained pegasus checkpoints are the same besides three attributes: :obj:`tokenizer.model_max_length` (maximum
  input size), :obj:`max_length` (the maximum number of tokens to generate) and :obj:`length_penalty`.
- The code to convert checkpoints trained in the author's `repo <https://github.com/google-research/pegasus>`_ can be
  found in ``convert_pegasus_tf_to_pytorch.py``.


Usage Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    >>> from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    >>> import torch
    >>> src_text = [
    ...     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
    >>> ]

    >>> model_name = 'google/pegasus-xsum'
    >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
    >>> tokenizer = PegasusTokenizer.from_pretrained(model_name)
    >>> model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    >>> batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
    >>> translated = model.generate(**batch)
    >>> tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    >>> assert tgt_text[0] == "California's largest electricity provider has turned off power to hundreds of thousands of customers."



PegasusConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PegasusConfig


PegasusTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

warning: ``add_tokens`` does not work at the moment.

.. autoclass:: transformers.PegasusTokenizer
    :members:


PegasusTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PegasusTokenizerFast
    :members:


PegasusModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PegasusModel
    :members: forward


PegasusForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PegasusForConditionalGeneration
    :members: forward


PegasusForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PegasusForCausalLM
    :members: forward


TFPegasusModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPegasusModel
    :members: call


TFPegasusForConditionalGeneration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPegasusForConditionalGeneration
    :members: call
