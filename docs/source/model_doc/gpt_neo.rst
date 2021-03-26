.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

EleutherAI GPT Neo
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPTNeo model was released in the `EleutherAI/gpt-neo <https://github.com/EleutherAI/gpt-neo>`__ repository by Sid
black, and Stella Biderman. It is a GPT2 like causal language model trained on the `Pile <https://pile.eleuther.ai/>`__
dataset.

The architecture is similar to GPT2 except the GPT Neo model uses local attention <> in every other layer with 256
window size so a token can only attend to last 256 positions.

Generation
_______________________________________________________________________________________________________________________

The:obj:`generate()` method can be used to generate text using GPT Neo model.

.. code-block::

    >>> from transformers import GPTNeoForCausalLM, GPTNeoTokenizer
    >>> model = GPTNeoForCausalLM.from_pretrained("eleutherai/gpt_neo_xl")
    >>> tokenizer = GPTNeoTokenizer.from_pretrained("eleutherai/gpt_neo_xl")

    >>> prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
    ...       "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
    ...       "researchers was the fact that the unicorns spoke perfect English."

    >>> input_ids = tokenizer(unicorns, return_tensors="pt").input_ids

    >>> gen_tokens = model.generate(ids, do_sample=True, temperature=0.9, max_length=100,)
    >>> gen_text = tokenizer.batch_decode(gen_tokens)[0]


GPTNeoConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoConfig
    :members:


GPTNeoTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


GPTNeoTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoTokenizerFast
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


GPTNeoModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoModel
    :members: forward


GPTNeoForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoForCausalLM
    :members: forward
