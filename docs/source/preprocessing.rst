.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

reprocessing data
=======================================================================================================================

In this tutorial, we'll explore how to preprocess your data using 🤗 Transformers. The main tool for this is what we
call a :doc:`tokenizer <main_classes/tokenizer>`. You can build one using the tokenizer class associated to the model
you would like to use, or directly with the :class:`~transformers.AutoTokenizer` class.

As we saw in the :doc:`quicktour </quicktour>`, the tokenizer will first split a given text in words (or part of words,
punctuation symbols, etc.) usually called `tokens`. Then it will convert those `tokens` into numbers, to be able to
build a tensor out of them and feed them to the model. It will also add any additional inputs the model might expect to
work properly.

.. note::

    If you plan on using a pretrained model, it's important to use the associated pretrained tokenizer: it will split
    the text you give it in tokens the same way for the pretraining corpus, and it will use the same correspondence
    token to index (that we usually call a `vocab`) as during pretraining.

To automatically download the vocab used during pretraining or fine-tuning a given model, you can use the
:func:`~transformers.AutoTokenizer.from_pretrained` method:

.. code-block::

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

Base use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~transformers.PreTrainedTokenizer` has many methods, but the only one you need to remember for preprocessing
is its ``__call__``: you just need to feed your sentence to your tokenizer object.

.. code-block::

    >>> encoded_input = tokenizer("Hello, I'm a single sentence!")
    >>> print(encoded_input)
    {'input_ids': [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102], 
     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

This returns a dictionary string to list of ints. The `input_ids <glossary.html#input-ids>`__ are the indices
corresponding to each token in our sentence. We will see below what the `attention_mask
<glossary.html#attention-mask>`__ is used for and in :ref:`the next section <sentence-pairs>` the goal of
`token_type_ids <glossary.html#token-type-ids>`__.

The tokenizer can decode a list of token ids in a proper sentence:

.. code-block::

    >>> tokenizer.decode(encoded_input["input_ids"])
    "[CLS] Hello, I'm a single sentence! [SEP]"

As you can see, the tokenizer automatically added some special tokens that the model expects. Not all models need
special tokens; for instance, if we had used `gpt2-medium` instead of `bert-base-cased` to create our tokenizer, we
would have seen the same sentence as the original one here. You can disable this behavior (which is only advised if you
have added those special tokens yourself) by passing ``add_special_tokens=False``.

If you have several sentences you want to process, you can do this efficiently by sending them as a list to the
tokenizer:

.. code-block::

    >>> batch_sentences = ["Hello I'm a single sentence",
    ...                    "And another sentence",
    ...                    "And the very very last one"]
    >>> encoded_inputs = tokenizer(batch_sentences)
    >>> print(encoded_inputs)
    {'input_ids': [[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
                   [101, 1262, 1330, 5650, 102],
                   [101, 1262, 1103, 1304, 1304, 1314, 1141, 102]],
     'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]],
     'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1]]}

We get back a dictionary once again, this time with values being lists of lists of ints.

If the purpose of sending several sentences at a time to the tokenizer is to build a batch to feed the model, you will
probably want:

- To pad each sentence to the maximum length there is in your batch.
- To truncate each sentence to the maximum length the model can accept (if applicable).
- To return tensors.

You can do all of this by using the following options when feeding your list of sentences to the tokenizer:

.. code-block::

    >>> ## PYTORCH CODE
    >>> batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    >>> print(batch)
    {'input_ids': tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
                          [ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
                          [ 101, 1262, 1103, 1304, 1304, 1314, 1141,  102,    0]]),
     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 0]])}
    >>> ## TENSORFLOW CODE
    >>> batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
    >>> print(batch)
    {'input_ids': tf.Tensor([[ 101, 8667,  146,  112,  182,  170, 1423, 5650,  102],
                          [ 101, 1262, 1330, 5650,  102,    0,    0,    0,    0],
                          [ 101, 1262, 1103, 1304, 1304, 1314, 1141,  102,    0]]),
     'token_type_ids': tf.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
     'attention_mask': tf.Tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 0, 0, 0, 0],
                               [1, 1, 1, 1, 1, 1, 1, 1, 0]])}

It returns a dictionary with string keys and tensor values. We can now see what the `attention_mask
<glossary.html#attention-mask>`__ is all about: it points out which tokens the model should pay attention to and which
ones it should not (because they represent padding in this case).


Note that if your model does not have a maximum length associated to it, the command above will throw a warning. You
can safely ignore it. You can also pass ``verbose=False`` to stop the tokenizer to throw those kinds of warnings.

.. _sentence-pairs:

Preprocessing pairs of sentences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you need to feed a pair of sentences to your model. For instance, if you want to classify if two sentences in
a pair are similar, or for question-answering models, which take a context and a question. For BERT models, the input
is then represented like this: :obj:`[CLS] Sequence A [SEP] Sequence B [SEP]`

You can encode a pair of sentences in the format expected by your model by supplying the two sentences as two arguments
(not a list since a list of two sentences will be interpreted as a batch of two single sentences, as we saw before).
This will once again return a dict string to list of ints:

.. code-block::

    >>> encoded_input = tokenizer("How old are you?", "I'm 6 years old")
    >>> print(encoded_input)
    {'input_ids': [101, 1731, 1385, 1132, 1128, 136, 102, 146, 112, 182, 127, 1201, 1385, 102], 
     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

This shows us what the `token_type_ids <glossary.html#token-type-ids>`__ are for: they indicate to the model which part
of the inputs correspond to the first sentence and which part corresponds to the second sentence. Note that
`token_type_ids` are not required or handled by all models. By default, a tokenizer will only return the inputs that
its associated model expects. You can force the return (or the non-return) of any of those special arguments by using
``return_input_ids`` or ``return_token_type_ids``.

If we decode the token ids we obtained, we will see that the special tokens have been properly added.

.. code-block::

    >>> tokenizer.decode(encoded_input["input_ids"])
    "[CLS] How old are you? [SEP] I'm 6 years old [SEP]"

If you have a list of pairs of sequences you want to process, you should feed them as two lists to your tokenizer: the
list of first sentences and the list of second sentences:

.. code-block::

    >>> batch_sentences = ["Hello I'm a single sentence",
    ...                    "And another sentence",
    ...                    "And the very very last one"]
    >>> batch_of_second_sentences = ["I'm a sentence that goes with the first sentence",
    ...                              "And I should be encoded with the second sentence",
    ...                              "And I go with the very last one"]
    >>> encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
    >>> print(encoded_inputs)
    {'input_ids': [[101, 8667, 146, 112, 182, 170, 1423, 5650, 102, 146, 112, 182, 170, 5650, 1115, 2947, 1114, 1103, 1148, 5650, 102], 
                   [101, 1262, 1330, 5650, 102, 1262, 146, 1431, 1129, 12544, 1114, 1103, 1248, 5650, 102], 
                   [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 1262, 146, 1301, 1114, 1103, 1304, 1314, 1141, 102]], 
    'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 
    'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

As we can see, it returns a dictionary where each value is a list of lists of ints.

To double-check what is fed to the model, we can decode each list in `input_ids` one by one:

.. code-block::

    >>> for ids in encoded_inputs["input_ids"]:
    >>>     print(tokenizer.decode(ids))
    [CLS] Hello I'm a single sentence [SEP] I'm a sentence that goes with the first sentence [SEP]
    [CLS] And another sentence [SEP] And I should be encoded with the second sentence [SEP]
    [CLS] And the very very last one [SEP] And I go with the very last one [SEP]

Once again, you can automatically pad your inputs to the maximum sentence length in the batch, truncate to the maximum
length the model can accept and return tensors directly with the following:

.. code-block::

    ## PYTORCH CODE
    batch = tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors="pt")
    ## TENSORFLOW CODE
    batch = tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors="tf")

Everything you always wanted to know about padding and truncation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have seen the commands that will work for most cases (pad your batch to the length of the maximum sentence and

truncate to the maximum length the mode can accept). However, the API supports more strategies if you need them. The
three arguments you need to know for this are :obj:`padding`, :obj:`truncation` and :obj:`max_length`.

- :obj:`padding` controls the padding. It can be a boolean or a string which should be:

    - :obj:`True` or :obj:`'longest'` to pad to the longest sequence in the batch (doing no padding if you only provide
      a single sequence).
    - :obj:`'max_length'` to pad to a length specified by the :obj:`max_length` argument or the maximum length accepted
      by the model if no :obj:`max_length` is provided (``max_length=None``). If you only provide a single sequence,
      padding will still be applied to it.
    - :obj:`False` or :obj:`'do_not_pad'` to not pad the sequences. As we have seen before, this is the default
      behavior.

- :obj:`truncation` controls the truncation. It can be a boolean or a string which should be:

    - :obj:`True` or :obj:`'only_first'` truncate to a maximum length specified by the :obj:`max_length` argument or
      the maximum length accepted by the model if no :obj:`max_length` is provided (``max_length=None``). This will
      only truncate the first sentence of a pair if a pair of sequence (or a batch of pairs of sequences) is provided.
    - :obj:`'only_second'` truncate to a maximum length specified by the :obj:`max_length` argument or the maximum
      length accepted by the model if no :obj:`max_length` is provided (``max_length=None``). This will only truncate
      the second sentence of a pair if a pair of sequence (or a batch of pairs of sequences) is provided.
    - :obj:`'longest_first'` truncate to a maximum length specified by the :obj:`max_length` argument or the maximum
      length accepted by the model if no :obj:`max_length` is provided (``max_length=None``). This will truncate token
      by token, removing a token from the longest sequence in the pair until the proper length is reached.
    - :obj:`False` or :obj:`'do_not_truncate'` to not truncate the sequences. As we have seen before, this is the
      default behavior.

- :obj:`max_length` to control the length of the padding/truncation. It can be an integer or :obj:`None`, in which case
  it will default to the maximum length the model can accept. If the model has no specific maximum input length,
  truncation/padding to :obj:`max_length` is deactivated.

Here is a table summarizing the recommend way to setup padding and truncation. If you use pair of inputs sequence in
any of the following examples, you can replace :obj:`truncation=True` by a :obj:`STRATEGY` selected in
:obj:`['only_first', 'only_second', 'longest_first']`, i.e. :obj:`truncation='only_second'` or :obj:`truncation=
'longest_first'` to control how both sequence in the pair are truncated as detailed before.

+--------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------+
| Truncation                           | Padding                           | Instruction                                                                                 |
+======================================+===================================+=============================================================================================+
| no truncation                        | no padding                        | :obj:`tokenizer(batch_sentences)`                                                           |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to max sequence in batch  | :obj:`tokenizer(batch_sentences, padding=True)` or                                          |
|                                      |                                   | :obj:`tokenizer(batch_sentences, padding='longest')`                                        |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to max model input length | :obj:`tokenizer(batch_sentences, padding='max_length')`                                     |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to specific length        | :obj:`tokenizer(batch_sentences, padding='max_length', max_length=42)`                      |
+--------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------+
| truncation to max model input length | no padding                        | :obj:`tokenizer(batch_sentences, truncation=True)` or                                       |
|                                      |                                   | :obj:`tokenizer(batch_sentences, truncation=STRATEGY)`                                      |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to max sequence in batch  | :obj:`tokenizer(batch_sentences, padding=True, truncation=True)` or                         |
|                                      |                                   | :obj:`tokenizer(batch_sentences, padding=True, truncation=STRATEGY)`                        |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to max model input length | :obj:`tokenizer(batch_sentences, padding='max_length', truncation=True)` or                 |
|                                      |                                   | :obj:`tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY)`                |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to specific length        | Not possible                                                                                |
+--------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------+
| truncation to specific length        | no padding                        | :obj:`tokenizer(batch_sentences, truncation=True, max_length=42)` or                        |
|                                      |                                   | :obj:`tokenizer(batch_sentences, truncation=STRATEGY, max_length=42)`                       |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to max sequence in batch  | :obj:`tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` or          |
|                                      |                                   | :obj:`tokenizer(batch_sentences, padding=True, truncation=STRATEGY, max_length=42)`         |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to max model input length | Not possible                                                                                |
|                                      +-----------------------------------+---------------------------------------------------------------------------------------------+
|                                      | padding to specific length        | :obj:`tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` or  |
|                                      |                                   | :obj:`tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY, max_length=42)` |
+--------------------------------------+-----------------------------------+---------------------------------------------------------------------------------------------+

Pre-tokenized inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tokenizer also accept pre-tokenized inputs. This is particularly useful when you want to compute labels and extract
predictions in `named entity recognition (NER) <https://en.wikipedia.org/wiki/Named-entity_recognition>`__ or
`part-of-speech tagging (POS tagging) <https://en.wikipedia.org/wiki/Part-of-speech_tagging>`__.

.. warning::

    Pre-tokenized does not mean your inputs are already tokenized (you wouldn't need to pass them through the tokenizer
    if that was the case) but just split into words (which is often the first step in subword tokenization algorithms
    like BPE).

If you want to use pre-tokenized inputs, just set :obj:`is_split_into_words=True` when passing your inputs to the
tokenizer. For instance, we have:

.. code-block::

    >>> encoded_input = tokenizer(["Hello", "I'm", "a", "single", "sentence"], is_split_into_words=True)
    >>> print(encoded_input)
    {'input_ids': [101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

Note that the tokenizer still adds the ids of special tokens (if applicable) unless you pass
``add_special_tokens=False``.

This works exactly as before for batch of sentences or batch of pairs of sentences. You can encode a batch of sentences
like this:

.. code-block::

    batch_sentences = [["Hello", "I'm", "a", "single", "sentence"],
                       ["And", "another", "sentence"],
                       ["And", "the", "very", "very", "last", "one"]]
    encoded_inputs = tokenizer(batch_sentences, is_split_into_words=True)

or a batch of pair sentences like this:

.. code-block::

    batch_of_second_sentences = [["I'm", "a", "sentence", "that", "goes", "with", "the", "first", "sentence"],
                                 ["And", "I", "should", "be", "encoded", "with", "the", "second", "sentence"],
                                 ["And", "I", "go", "with", "the", "very", "last", "one"]]
    encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences, is_split_into_words=True)

And you can add padding, truncation as well as directly return tensors like before:

.. code-block::

    ## PYTORCH CODE
    batch = tokenizer(batch_sentences,
                      batch_of_second_sentences,
                      is_split_into_words=True,
                      padding=True,
                      truncation=True,
                      return_tensors="pt")
    ## TENSORFLOW CODE
    batch = tokenizer(batch_sentences,
                      batch_of_second_sentences,
                      is_split_into_words=True,
                      padding=True,
                      truncation=True,
                      return_tensors="tf")
