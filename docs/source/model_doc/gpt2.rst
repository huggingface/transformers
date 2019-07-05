OpenAI GPT2
----------------------------------------------------


``GPT2Tokenizer``
~~~~~~~~~~~~~~~~~~~~~

``GPT2Tokenizer`` perform byte-level Byte-Pair-Encoding (BPE) tokenization.

This class has three arguments:


* ``vocab_file``\ : path to a vocabulary file.
* ``merges_file``\ : path to a file containing the BPE merges.
* ``errors``\ : How to handle unicode decoding errors. **Default = ``replace``\ **

and two methods:


* ``tokenize(text)``\ : convert a ``str`` in a list of ``str`` tokens by performing byte-level BPE.
* ``convert_tokens_to_ids(tokens)``\ : convert a list of ``str`` tokens in a list of ``int`` indices in the vocabulary.
* ``convert_ids_to_tokens(tokens)``\ : convert a list of ``int`` indices in a list of ``str`` tokens in the vocabulary.
* ``set_special_tokens(self, special_tokens)``\ : update the list of special tokens (see above arguments)
* ``encode(text)``\ : convert a ``str`` in a list of ``int`` tokens by performing byte-level BPE.
* ``decode(tokens)``\ : convert back a list of ``int`` tokens in a ``str``.
* `save_vocabulary(directory_path)`: save the vocabulary, merge and special tokens files to `directory_path`. Return the path to the three files: ``vocab_file_path``\ , ``merge_file_path``\ , ``special_tokens_file_path``. The vocabulary can be reloaded with ``OpenAIGPTTokenizer.from_pretrained('directory_path')``.

Please refer to `\ ``tokenization_gpt2.py`` <./pytorch_pretrained_bert/tokenization_gpt2.py>`_ for more details on the ``GPT2Tokenizer``.


14. ``GPT2Model``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.GPT2Model
    :members:


15. ``GPT2LMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.GPT2LMHeadModel
    :members:


16. ``GPT2DoubleHeadsModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.GPT2DoubleHeadsModel
    :members:
