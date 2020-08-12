Glossary
^^^^^^^^

General terms
-------------

- autoencoding models: see MLM
- autoregressive models: see CLM
- CLM: causal language modeling, a pretraining task where the model reads the texts in order and has to predict the
  next word. It's usually done by reading the whole sentence but using a mask inside the model to hide the future 
  tokens at a certain timestep.
- MLM: masked language modeling, a pretraining task where the model sees a corrupted version of the texts, usually done
  by masking some tokens randomly, and has to predict the original text.
- multimodal: a task that combines texts with another kind of inputs (for instance images).
- NLG: natural language generation, all tasks related to generating text ( for instance talk with transformers,
  translation)
- NLP: natural language processing, a generic way to say "deal with texts".
- NLU: natural language understanding, all tasks related to understanding what is in a text (for instance classifying
  the whole text, individual words)
- pretrained model: a model that has been pretrained on some data (for instance all of Wikipedia). Pretraining methods
  involve a self-supervised objective, which can be reading the text and trying to predict the next word (see CLM) or 
  masking some words and trying to predict them (see MLM).
- RNN: recurrent neural network, a type of model that uses a loop over a layer to process texts.
- seq2seq or sequence-to-sequence: models that generate a new sequence from an input, like translation models, or
  summarization models (such as :doc:`Bart </model_doc/bart>` or :doc:`T5 </model_doc/t5>`).
- token: a part of a sentence, usually a word, but can also be a subword (non-common words are often split in subwords)
  or a punctuation symbol.

Model inputs
------------

Every model is different yet bears similarities with the others. Therefore most models use the same inputs, which are
detailed here alongside usage examples.

.. _input-ids:

Input IDs
~~~~~~~~~

The input ids are often the only required parameters to be passed to the model as input. *They are token indices,
numerical representations of tokens building the sequences that will be used as input by the model*.

Each tokenizer works differently but the underlying mechanism remains the same. Here's an example using the BERT
tokenizer, which is a `WordPiece <https://arxiv.org/pdf/1609.08144.pdf>`__ tokenizer:

::

    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    >>> sequence = "A Titan RTX has 24GB of VRAM"

The tokenizer takes care of splitting the sequence into tokens available in the tokenizer vocabulary.

::

    >>> tokenized_sequence = tokenizer.tokenize(sequence)

The tokens are either words or subwords. Here for instance, "VRAM" wasn't in the model vocabulary, so it's been split
in "V", "RA" and "M". To indicate those tokens are not separate words but parts of the same word, a double-dash is
added for "RA" and "M":

::

    >>> print(tokenized_sequence)
    ['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']

These tokens can then be converted into IDs which are understandable by the model. This can be done by directly feeding
the sentence to the tokenizer, which leverages the Rust implementation of
`huggingface/tokenizers <https://github.com/huggingface/tokenizers>`__ for peak performance.

::

    >>> encoded_sequence = tokenizer(sequence)["input_ids"]

The tokenizer returns a dictionary with all the arguments necessary for its corresponding model to work properly. The
token indices are under the key "input_ids":

::

    >>> print(encoded_sequence)
    [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]

Note that the tokenizer automatically adds "special tokens" (if the associated model rely on them) which are special
IDs the model sometimes uses. If we decode the previous sequence of ids,

::

    >>> decoded_sequence = tokenizer.decode(encoded_sequence)

we will see 

::

    >>> print(decoded_sequence)
    [CLS] A Titan RTX has 24GB of VRAM [SEP]

because this is the way a :class:`~transformers.BertModel` is going to expect its inputs.

.. _attention-mask:

Attention mask
~~~~~~~~~~~~~~

The attention mask is an optional argument used when batching sequences together. This argument indicates to the
model which tokens should be attended to, and which should not.

For example, consider these two sequences:

::

    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    >>> sequence_a = "This is a short sequence."
    >>> sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

    >>> encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
    >>> encoded_sequence_b = tokenizer(sequence_b)["input_ids"]

The encoded versions have different lengths:

::

    >>> len(encoded_sequence_a), len(encoded_sequence_b)
    (8, 19)

Therefore, we can't be put then together in a same tensor as-is. The first sequence needs to be padded up to the length
of the second one, or the second one needs to be truncated down to the length of the first one.

In the first case, the list of IDs will be extended by the padding indices. We can pass a list to the tokenizer and ask
it to pad like this:

::

    >>> padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)

We can see that 0s have been added on the right of the first sentence to make it the same length as the second one:

::

    >>> padded_sequences["input_ids"]
    [[101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]]

This can then be converted into a tensor in PyTorch or TensorFlow. The attention mask is a binary tensor indicating
the position of the padded indices so that the model does not attend to them. For the
:class:`~transformers.BertTokenizer`, :obj:`1` indicate a value that should be attended to while :obj:`0` indicate
a padded value. This attention mask is in the dictionary returned by the tokenizer under the key "attention_mask":

::

    >>> padded_sequences["attention_mask"]
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

.. _token-type-ids:

Token Type IDs
~~~~~~~~~~~~~~

Some models' purpose is to do sequence classification or question answering. These require two different sequences to
be encoded in the same input IDs. They are usually separated by special tokens, such as the classifier and separator
tokens. For example, the BERT model builds its two sequence input as such:

::

   >>> # [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]

We can use our tokenizer to automatically generate such a sentence by passing the two sequences as two arguments (and
not a list like before) like this:

::

    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    >>> sequence_a = "HuggingFace is based in NYC"
    >>> sequence_b = "Where is HuggingFace based?"

    >>> encoded_dict = tokenizer(sequence_a, sequence_b)
    >>> decoded = tokenizer.decode(encoded_dict["input_ids"])

which will return:

::

    >>> print(decoded)
    [CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]

This is enough for some models to understand where one sequence ends and where another begins. However, other models
such as BERT have an additional mechanism, which are the token type IDs (also called segment IDs). They are a binary
mask identifying the different sequences in the model.

The tokenizer returns in the dictionary under the key "token_type_ids":

::

    >>> encoded_dict['token_type_ids']
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

The first sequence, the "context" used for the question, has all its tokens represented by :obj:`0`, whereas the
question has all its tokens represented by :obj:`1`. Some models, like :class:`~transformers.XLNetModel` use an
additional token represented by a :obj:`2`.

.. _position-ids:

Position IDs
~~~~~~~~~~~~

The position IDs are used by the model to identify which token is at which position. Contrary to RNNs that have the
position of each token embedded within them, transformers are unaware of the position of each token. The position
IDs are created for this purpose.

They are an optional parameter. If no position IDs are passed to the model, they are automatically created as absolute
positional embeddings.

Absolute positional embeddings are selected in the range ``[0, config.max_position_embeddings - 1]``. Some models
use other types of positional embeddings, such as sinusoidal position embeddings or relative position embeddings.

.. _feed-forward-chunking:

Feed Forward Chunking
~~~~~~~~~~~~~~~~~~~~~

In transformers two feed forward layers usually follows the self attention layer in each residual attention block.
The intermediate embedding size of the feed forward layers is often bigger than the hidden size of the model (e.g.,
for ``bert-base-uncased``). 

For an input of size ``[batch_size, sequence_length]``, the memory required to store the intermediate feed forward
embeddings ``[batch_size, sequence_length, config.intermediate_size]`` can account for a large fraction of the memory
use. The authors of `Reformer: The Efficient Transformer <https://arxiv.org/abs/2001.04451>`_ noticed that since the
computation is independent of the ``sequence_length`` dimension, it is mathematically equivalent to compute the output
embeddings of both feed forward layers ``[batch_size, config.hidden_size]_0, ..., [batch_size, config.hidden_size]_n`` 
individually and concat them afterward to ``[batch_size, sequence_length, config.hidden_size]`` with
``n = sequence_length``, which trades increased computation time against reduced memory use, but yields a
mathematically **equivalent** result.

For models employing the function :func:`~.transformers.apply_chunking_to_forward`, the ``chunk_size`` defines the
number of output embeddings that are computed in parallel and thus defines the trade-off between memory and time
complexity.  If ``chunk_size`` is set to 0, no feed forward chunking is done.
