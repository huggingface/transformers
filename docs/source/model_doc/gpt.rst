OpenAI GPT
----------------------------------------------------


``OpenAIGPTTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``OpenAIGPTTokenizer`` perform Byte-Pair-Encoding (BPE) tokenization.

This class has four arguments:


* ``vocab_file``\ : path to a vocabulary file.
* ``merges_file``\ : path to a file containing the BPE merges.
* ``max_len``\ : max length to filter the input of the Transformer. Default to pre-trained value for the model if ``None``. **Default = None**
* ``special_tokens``\ : a list of tokens to add to the vocabulary for fine-tuning. If SpaCy is not installed and BERT's ``BasicTokenizer`` is used as the pre-BPE tokenizer, these tokens are not split. **Default= None**

and five methods:


* ``tokenize(text)``\ : convert a ``str`` in a list of ``str`` tokens by performing BPE tokenization.
* ``convert_tokens_to_ids(tokens)``\ : convert a list of ``str`` tokens in a list of ``int`` indices in the vocabulary.
* ``convert_ids_to_tokens(tokens)``\ : convert a list of ``int`` indices in a list of ``str`` tokens in the vocabulary.
* ``set_special_tokens(self, special_tokens)``\ : update the list of special tokens (see above arguments)
* ``encode(text)``\ : convert a ``str`` in a list of ``int`` tokens by performing BPE encoding.
* `decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)`: decode a list of `int` indices in a string and do some post-processing if needed: (i) remove special tokens from the output and (ii) clean up tokenization spaces.
* `save_vocabulary(directory_path)`: save the vocabulary, merge and special tokens files to `directory_path`. Return the path to the three files: ``vocab_file_path``\ , ``merge_file_path``\ , ``special_tokens_file_path``. The vocabulary can be reloaded with ``OpenAIGPTTokenizer.from_pretrained('directory_path')``.

Please refer to the doc strings and code in `\ ``tokenization_openai.py`` <./pytorch_pretrained_bert/tokenization_openai.py>`_ for the details of the ``OpenAIGPTTokenizer``.


``OpenAIAdam``
~~~~~~~~~~~~~~~~~~

``OpenAIAdam`` is similar to ``BertAdam``.
The differences with ``BertAdam`` is that ``OpenAIAdam`` compensate for bias as in the regular Adam optimizer.

``OpenAIAdam`` accepts the same arguments as ``BertAdam``.


9. ``OpenAIGPTModel``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.OpenAIGPTModel
    :members:


10. ``OpenAIGPTLMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.OpenAIGPTLMHeadModel
    :members:


11. ``OpenAIGPTDoubleHeadsModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.OpenAIGPTDoubleHeadsModel
    :members:
