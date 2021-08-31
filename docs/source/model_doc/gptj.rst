.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

GPT-J
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPT-J model was released in the `kingoflolz/mesh-transformer-jax
<https://github.com/kingoflolz/mesh-transformer-jax>`__ repository by Ben Wang and Aran Komatsuzaki. It is a GPT-2-like
causal language model trained on `the Pile <https://pile.eleuther.ai/>`__ dataset.

This model was contributed by `Stella Biderman <https://huggingface.co/stellaathena>`__.

Tips:

- Running [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B) in float32 precision on GPU requires at least 24 GB of
  RAM. On GPUs with less than 24 GB RAM, one should therefore load the model in half-precision:

.. code-block::

    >>> from transformers import GPTJForCausalLM
    >>> import torch

    >>> model =  GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)

Generation
_______________________________________________________________________________________________________________________

The :meth:`~transformers.generation_utils.GenerationMixin.generate` method can be used to generate text using GPT-J
model.

.. code-block::

    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    >>> prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
    ...          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
    ...          "researchers was the fact that the unicorns spoke perfect English."

    >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    >>> gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    >>> gen_text = tokenizer.batch_decode(gen_tokens)[0]

...or in float16 precision:

.. code-block::

    >>> from transformers import GPTJForCausalLM, AutoTokenizer
    >>> import torch

    >>> model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    >>> prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
    ...          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
    ...          "researchers was the fact that the unicorns spoke perfect English."

    >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    >>> gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    >>> gen_text = tokenizer.batch_decode(gen_tokens)[0]


GPTJConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJConfig
    :members:

GPTJModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJModel
    :members: forward


GPTJForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJForCausalLM
    :members: forward


GPTJForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJForSequenceClassification
    :members: forward
