Glossary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every model is different yet bears similarities with the others. Therefore most models use the same inputs, which are
detailed here alongside usage examples.

Input IDs
--------------------------

The input ids are often the only required parameters to be passed to the model as input. *They are token indices,
numerical representations of tokens building the sequences that will be used as input by the model*.

Each tokenizer works differently but the underlying mechanism remains the same. Here's an example using the BERT
tokenizer, which is a `WordPiece <https://arxiv.org/pdf/1609.08144.pdf>`__ tokenizer:

::

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    sequence = "A Titan RTX has 24GB of VRAM"

The tokenizer takes care of splitting the sequence into tokens available in the tokenizer vocabulary.

::

    # Continuation of the previous script
    tokenized_sequence = tokenizer.tokenize(sequence)
    assert tokenized_sequence == ['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']

These tokens can then be converted into IDs which are understandable by the model. Several methods are available for
this, the recommended being `encode` or `encode_plus`, which leverage the Rust implementation of
`huggingface/tokenizers <https://github.com/huggingface/tokenizers>`__ for peak performance.

::

    # Continuation of the previous script
    encoded_sequence = tokenizer.encode(sequence)
    assert encoded_sequence == [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]

The `encode` and `encode_plus` methods automatically add "special tokens" which are special IDs the model uses.

Attention mask
--------------------------

The attention mask is an optional argument used when batching sequences together. This argument indicates to the
model which tokens should be attended to, and which should not.

For example, consider these two sequences:

::

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    sequence_a = "This is a short sequence."
    sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

    encoded_sequence_a = tokenizer.encode(sequence_a)
    assert len(encoded_sequence_a) == 8

    encoded_sequence_b = tokenizer.encode(sequence_b)
    assert len(encoded_sequence_b) == 19

These two sequences have different lengths and therefore can't be put together in a same tensor as-is. The first
sequence needs to be padded up to the length of the second one, or the second one needs to be truncated down to
the length of the first one.

In the first case, the list of IDs will be extended by the padding indices:

::

    # Continuation of the previous script
    padded_sequence_a = tokenizer.encode(sequence_a, max_length=19, pad_to_max_length=True)

    assert padded_sequence_a == [101, 1188, 1110, 170, 1603, 4954,  119, 102,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,   0]
    assert encoded_sequence_b == [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]

These can then be converted into a tensor in PyTorch or TensorFlow. The attention mask is a binary tensor indicating
the position of the padded indices so that the model does not attend to them. For the
:class:`~transformers.BertTokenizer`, :obj:`1` indicate a value that should be attended to while :obj:`0` indicate
a padded value.

The method :func:`~transformers.PreTrainedTokenizer.encode_plus` may be used to obtain the attention mask directly:

::

    # Continuation of the previous script
    sequence_a_dict = tokenizer.encode_plus(sequence_a, max_length=19, pad_to_max_length=True)

    assert sequence_a_dict['input_ids'] == [101, 1188, 1110, 170, 1603, 4954, 119, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert sequence_a_dict['attention_mask'] == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


Token Type IDs
--------------------------

Some models' purpose is to do sequence classification or question answering. These require two different sequences to
be encoded in the same input IDs. They are usually separated by special tokens, such as the classifier and separator
tokens. For example, the BERT model builds its two sequence input as such:

::

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # [CLS] SEQ_A [SEP] SEQ_B [SEP]

    sequence_a = "HuggingFace is based in NYC"
    sequence_b = "Where is HuggingFace based?"

    encoded_sequence = tokenizer.encode(sequence_a, sequence_b)
    assert tokenizer.decode(encoded_sequence) == "[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]"

This is enough for some models to understand where one sequence ends and where another begins. However, other models
such as BERT have an additional mechanism, which are the segment IDs. The Token Type IDs are a binary mask identifying
the different sequences in the model.

We can leverage :func:`~transformers.PreTrainedTokenizer.encode_plus` to output the Token Type IDs for us:

::

    # Continuation of the previous script
    encoded_dict = tokenizer.encode_plus(sequence_a, sequence_b)

    assert encoded_dict['input_ids'] == [101, 20164, 10932, 2271, 7954, 1110, 1359, 1107, 17520, 102, 2777, 1110, 20164, 10932, 2271, 7954, 1359, 136, 102]
    assert encoded_dict['token_type_ids'] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

The first sequence, the "context" used for the question, has all its tokens represented by :obj:`0`, whereas the
question has all its tokens represented by :obj:`1`. Some models, like :class:`~transformers.XLNetModel` use an
additional token represented by a :obj:`2`.


Position IDs
--------------------------

The position IDs are used by the model to identify which token is at which position. Contrary to RNNs that have the
position of each token embedded within them, transformers are unaware of the position of each token. The position
IDs are created for this purpose.

They are an optional parameter. If no position IDs are passed to the model, they are automatically created as absolute
positional embeddings.

Absolute positional embeddings are selected in the range ``[0, config.max_position_embeddings - 1]``. Some models
use other types of positional embeddings, such as sinusoidal position embeddings or relative position embeddings.


Feed Forward Chunking
--------------------------

In transformers two feed forward layers usually follows the self attention layer in each residual attention block. The intermediate embedding size of the feed forward layers is often bigger than the hidden size of the model (*e.g.* for ``bert-base-uncased``). 

For an input of size ``[batch_size, sequence_length]``, the memory required to store the intermediate feed forward embeddings ``[batch_size, sequence_length, config.intermediate_size]`` can account for a large fraction of the memory use. The authors of `Reformer: The Efficient Transformer <https://arxiv.org/abs/2001.04451>`_ noticed that since the computation is independent of the ``sequence_length`` dimension, it is mathematically equivalent to compute the output embeddings of both feed forward layers ``[batch_size, config.hidden_size]_0, ..., [batch_size, config.hidden_size]_n``  individually and concat them afterward to ``[batch_size, sequence_length, config.hidden_size]`` with ``n = sequence_length``, which trades increased computation time against reduced memory use, but yields a mathematically **equivalent** result.

For models employing the function :func:`~.transformers.apply_chunking_to_forward`, the ``chunk_size`` defines the number of output embeddings that are computed in parallel and thus defines the trade-off between memory and time complexity. 
If ``chunk_size`` is set to 0, no feed forward chunking is done.
