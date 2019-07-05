BERT
----------------------------------------------------

``BertTokenizer``
~~~~~~~~~~~~~~~~~~~~~

``BertTokenizer`` perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization.

This class has five arguments:


* ``vocab_file``\ : path to a vocabulary file.
* ``do_lower_case``\ : convert text to lower-case while tokenizing. **Default = True**.
* ``max_len``\ : max length to filter the input of the Transformer. Default to pre-trained value for the model if ``None``. **Default = None**
* ``do_basic_tokenize``\ : Do basic tokenization before wordpice tokenization. Set to false if text is pre-tokenized. **Default = True**.
* ``never_split``\ : a list of tokens that should not be splitted during tokenization. **Default = ``["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]``\ **

and three methods:


* ``tokenize(text)``\ : convert a ``str`` in a list of ``str`` tokens by (1) performing basic tokenization and (2) WordPiece tokenization.
* ``convert_tokens_to_ids(tokens)``\ : convert a list of ``str`` tokens in a list of ``int`` indices in the vocabulary.
* ``convert_ids_to_tokens(tokens)``\ : convert a list of ``int`` indices in a list of ``str`` tokens in the vocabulary.
* `save_vocabulary(directory_path)`: save the vocabulary file to `directory_path`. Return the path to the saved vocabulary file: ``vocab_file_path``. The vocabulary can be reloaded with ``BertTokenizer.from_pretrained('vocab_file_path')`` or ``BertTokenizer.from_pretrained('directory_path')``.

Please refer to the doc strings and code in `\ ``tokenization.py`` <./pytorch_pretrained_bert/tokenization.py>`_ for the details of the ``BasicTokenizer`` and ``WordpieceTokenizer`` classes. In general it is recommended to use ``BertTokenizer`` unless you know what you are doing.


``BertAdam``
~~~~~~~~~~~~~~~~

``BertAdam`` is a ``torch.optimizer`` adapted to be closer to the optimizer used in the TensorFlow implementation of Bert. The differences with PyTorch Adam optimizer are the following:


* BertAdam implements weight decay fix,
* BertAdam doesn't compensate for bias as in the regular Adam optimizer.

The optimizer accepts the following arguments:


* ``lr`` : learning rate
* ``warmup`` : portion of ``t_total`` for the warmup, ``-1``  means no warmup. Default : ``-1``
* ``t_total`` : total number of training steps for the learning
    rate schedule, ``-1``  means constant learning rate. Default : ``-1``
* ``schedule`` : schedule to use for the warmup (see above).
    Can be ``'warmup_linear'``\ , ``'warmup_constant'``\ , ``'warmup_cosine'``\ , ``'none'``\ , ``None`` or a ``_LRSchedule`` object (see below).
    If ``None`` or ``'none'``\ , learning rate is always kept constant.
    Default : ``'warmup_linear'``
* ``b1`` : Adams b1. Default : ``0.9``
* ``b2`` : Adams b2. Default : ``0.999``
* ``e`` : Adams epsilon. Default : ``1e-6``
* ``weight_decay:`` Weight decay. Default : ``0.01``
* ``max_grad_norm`` : Maximum norm for the gradients (\ ``-1`` means no clipping). Default : ``1.0``


1. ``BertModel``
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertModel
    :members:


2. ``BertForPreTraining``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertForPreTraining
    :members:


3. ``BertForMaskedLM``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertForMaskedLM
    :members:


4. ``BertForNextSentencePrediction``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertForNextSentencePrediction
    :members:


5. ``BertForSequenceClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertForSequenceClassification
    :members:


6. ``BertForMultipleChoice``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertForMultipleChoice
    :members:


7. ``BertForTokenClassification``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertForTokenClassification
    :members:


8. ``BertForQuestionAnswering``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.BertForQuestionAnswering
    :members:

