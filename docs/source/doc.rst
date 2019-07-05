Docs
================================================



Here is a detailed documentation of the classes in the package and how to use them:

.. list-table::
   :header-rows: 1

   * - Sub-section
     - Description
   * - `Loading pre-trained weights <#loading-google-ai-or-openai-pre-trained-weights-or-pytorch-dump>`_
     - How to load Google AI/OpenAI's pre-trained weight or a PyTorch saved instance
   * - `Serialization best-practices <#serialization-best-practices>`_
     - How to save and reload a fine-tuned model
   * - `Configurations <#configurations>`_
     - API of the configuration classes for BERT, GPT, GPT-2 and Transformer-XL
   * - `Models <#models>`_
     - API of the PyTorch model classes for BERT, GPT, GPT-2 and Transformer-XL
   * - `Tokenizers <#tokenizers>`_
     - API of the tokenizers class for BERT, GPT, GPT-2 and Transformer-XL
   * - `Optimizers <#optimizers>`_
     - API of the optimizers


Loading Google AI or OpenAI pre-trained weights or PyTorch dump
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``from_pretrained()`` method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To load one of Google AI's, OpenAI's pre-trained models or a PyTorch saved model (an instance of ``BertForPreTraining`` saved with ``torch.save()``\ ), the PyTorch model classes and the tokenizer can be instantiated using the ``from_pretrained()`` method:

.. code-block:: python

   model = BERT_CLASS.from_pretrained(PRE_TRAINED_MODEL_NAME_OR_PATH, cache_dir=None, from_tf=False, state_dict=None, *input, **kwargs)

where


* ``BERT_CLASS`` is either a tokenizer to load the vocabulary (\ ``BertTokenizer`` or ``OpenAIGPTTokenizer`` classes) or one of the eight BERT or three OpenAI GPT PyTorch model classes (to load the pre-trained weights): ``BertModel``\ , ``BertForMaskedLM``\ , ``BertForNextSentencePrediction``\ , ``BertForPreTraining``\ , ``BertForSequenceClassification``\ , ``BertForTokenClassification``\ , ``BertForMultipleChoice``\ , ``BertForQuestionAnswering``\ , ``OpenAIGPTModel``\ , ``OpenAIGPTLMHeadModel`` or ``OpenAIGPTDoubleHeadsModel``\ , and
*
  ``PRE_TRAINED_MODEL_NAME_OR_PATH`` is either:


  *
    the shortcut name of a Google AI's or OpenAI's pre-trained model selected in the list:


    * ``bert-base-uncased``: 12-layer, 768-hidden, 12-heads, 110M parameters
    * ``bert-large-uncased``: 24-layer, 1024-hidden, 16-heads, 340M parameters
    * ``bert-base-cased``: 12-layer, 768-hidden, 12-heads , 110M parameters
    * ``bert-large-cased``: 24-layer, 1024-hidden, 16-heads, 340M parameters
    * ``bert-base-multilingual-uncased``: (Orig, not recommended) 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    * ``bert-base-multilingual-cased``: **(New, recommended)** 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    * ``bert-base-chinese``: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
    * ``bert-base-german-cased``: Trained on German data only, 12-layer, 768-hidden, 12-heads, 110M parameters `Performance Evaluation <https://deepset.ai/german-bert>`_
    * ``bert-large-uncased-whole-word-masking``: 24-layer, 1024-hidden, 16-heads, 340M parameters - Trained with Whole Word Masking (mask all of the the tokens corresponding to a word at once)
    * ``bert-large-cased-whole-word-masking``: 24-layer, 1024-hidden, 16-heads, 340M parameters - Trained with Whole Word Masking (mask all of the the tokens corresponding to a word at once)
    * ``bert-large-uncased-whole-word-masking-finetuned-squad``: The ``bert-large-uncased-whole-word-masking`` model finetuned on SQuAD (using the ``run_bert_squad.py`` examples). Results: *exact_match: 86.91579943235573, f1: 93.1532499015869*
    * ``openai-gpt``: OpenAI GPT English model, 12-layer, 768-hidden, 12-heads, 110M parameters
    * ``gpt2``: OpenAI GPT-2 English model, 12-layer, 768-hidden, 12-heads, 117M parameters
    * ``gpt2-medium``: OpenAI GPT-2 English model, 24-layer, 1024-hidden, 16-heads, 345M parameters
    * ``transfo-xl-wt103``: Transformer-XL English model trained on wikitext-103, 18-layer, 1024-hidden, 16-heads, 257M parameters

  *
    a path or url to a pretrained model archive containing:


    * ``bert_config.json`` or ``openai_gpt_config.json`` a configuration file for the model, and
    * ``pytorch_model.bin`` a PyTorch dump of a pre-trained instance of ``BertForPreTraining``\ , ``OpenAIGPTModel``\ , ``TransfoXLModel``\ , ``GPT2LMHeadModel`` (saved with the usual ``torch.save()``\ )

  If ``PRE_TRAINED_MODEL_NAME_OR_PATH`` is a shortcut name, the pre-trained weights will be downloaded from AWS S3 (see the links `here <https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/pytorch_pretrained_bert/modeling.py>`_\ ) and stored in a cache folder to avoid future download (the cache folder can be found at ``~/.pytorch_pretrained_bert/``\ ).

*
  ``cache_dir`` can be an optional path to a specific directory to download and cache the pre-trained model weights. This option is useful in particular when you are using distributed training: to avoid concurrent access to the same weights you can set for example ``cache_dir='./pretrained_model_{}'.format(args.local_rank)`` (see the section on distributed training for more information).

* ``from_tf``\ : should we load the weights from a locally saved TensorFlow checkpoint
* ``state_dict``\ : an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
* ``*inputs``\ , `**kwargs`: additional input for the specific Bert class (ex: num_labels for BertForSequenceClassification)

``Uncased`` means that the text has been lowercased before WordPiece tokenization, e.g., ``John Smith`` becomes ``john smith``. The Uncased model also strips out any accent markers. ``Cased`` means that the true case and accent markers are preserved. Typically, the Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging). For information about the Multilingual and Chinese model, see the `Multilingual README <https://github.com/google-research/bert/blob/master/multilingual.md>`_ or the original TensorFlow repository.

When using an ``uncased model``\ , make sure to pass ``--do_lower_case`` to the example training scripts (or pass ``do_lower_case=True`` to FullTokenizer if you're using your own script and loading the tokenizer your-self.).

Examples:

.. code-block:: python

   # BERT
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

   # OpenAI GPT
   tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
   model = OpenAIGPTModel.from_pretrained('openai-gpt')

   # Transformer-XL
   tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
   model = TransfoXLModel.from_pretrained('transfo-xl-wt103')

   # OpenAI GPT-2
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   model = GPT2Model.from_pretrained('gpt2')

Cache directory
~~~~~~~~~~~~~~~

``pytorch_pretrained_bert`` save the pretrained weights in a cache directory which is located at (in this order of priority):


* ``cache_dir`` optional arguments to the ``from_pretrained()`` method (see above),
* shell environment variable ``PYTORCH_PRETRAINED_BERT_CACHE``\ ,
* PyTorch cache home + ``/pytorch_pretrained_bert/``
  where PyTorch cache home is defined by (in this order):

  * shell environment variable ``ENV_TORCH_HOME``
  * shell environment variable ``ENV_XDG_CACHE_HOME`` + ``/torch/``\ )
  * default: ``~/.cache/torch/``

Usually, if you don't set any specific environment variable, ``pytorch_pretrained_bert`` cache will be at ``~/.cache/torch/pytorch_pretrained_bert/``.

You can alsways safely delete ``pytorch_pretrained_bert`` cache but the pretrained model weights and vocabulary files wil have to be re-downloaded from our S3.

Serialization best-practices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section explain how you can save and re-load a fine-tuned model (BERT, GPT, GPT-2 and Transformer-XL).
There are three types of files you need to save to be able to reload a fine-tuned model:


* the model it-self which should be saved following PyTorch serialization `best practices <https://pytorch.org/docs/stable/notes/serialization.html#best-practices>`_\ ,
* the configuration file of the model which is saved as a JSON file, and
* the vocabulary (and the merges for the BPE-based models GPT and GPT-2).

The *default filenames* of these files are as follow:


* the model weights file: ``pytorch_model.bin``\ ,
* the configuration file: ``config.json``\ ,
* the vocabulary file: ``vocab.txt`` for BERT and Transformer-XL, ``vocab.json`` for GPT/GPT-2 (BPE vocabulary),
* for GPT/GPT-2 (BPE vocabulary) the additional merges file: ``merges.txt``.

**If you save a model using these *default filenames*\ , you can then re-load the model and tokenizer using the ``from_pretrained()`` method.**

Here is the recommended way of saving the model, configuration and vocabulary to an ``output_dir`` directory and reloading the model and tokenizer afterwards:

.. code-block:: python

   from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME

   output_dir = "./models/"

   # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

   # If we have a distributed model, save only the encapsulated model
   # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
   model_to_save = model.module if hasattr(model, 'module') else model

   # If we save using the predefined names, we can load using `from_pretrained`
   output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
   output_config_file = os.path.join(output_dir, CONFIG_NAME)

   torch.save(model_to_save.state_dict(), output_model_file)
   model_to_save.config.to_json_file(output_config_file)
   tokenizer.save_vocabulary(output_dir)

   # Step 2: Re-load the saved model and vocabulary

   # Example for a Bert model
   model = BertForQuestionAnswering.from_pretrained(output_dir)
   tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)  # Add specific options if needed
   # Example for a GPT model
   model = OpenAIGPTDoubleHeadsModel.from_pretrained(output_dir)
   tokenizer = OpenAIGPTTokenizer.from_pretrained(output_dir)

Here is another way you can save and reload the model if you want to use specific paths for each type of files:

.. code-block:: python

   output_model_file = "./models/my_own_model_file.bin"
   output_config_file = "./models/my_own_config_file.bin"
   output_vocab_file = "./models/my_own_vocab_file.bin"

   # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

   # If we have a distributed model, save only the encapsulated model
   # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
   model_to_save = model.module if hasattr(model, 'module') else model

   torch.save(model_to_save.state_dict(), output_model_file)
   model_to_save.config.to_json_file(output_config_file)
   tokenizer.save_vocabulary(output_vocab_file)

   # Step 2: Re-load the saved model and vocabulary

   # We didn't save using the predefined WEIGHTS_NAME, CONFIG_NAME names, we cannot load using `from_pretrained`.
   # Here is how to do it in this situation:

   # Example for a Bert model
   config = BertConfig.from_json_file(output_config_file)
   model = BertForQuestionAnswering(config)
   state_dict = torch.load(output_model_file)
   model.load_state_dict(state_dict)
   tokenizer = BertTokenizer(output_vocab_file, do_lower_case=args.do_lower_case)

   # Example for a GPT model
   config = OpenAIGPTConfig.from_json_file(output_config_file)
   model = OpenAIGPTDoubleHeadsModel(config)
   state_dict = torch.load(output_model_file)
   model.load_state_dict(state_dict)
   tokenizer = OpenAIGPTTokenizer(output_vocab_file)

Configurations
^^^^^^^^^^^^^^

Models (BERT, GPT, GPT-2 and Transformer-XL) are defined and build from configuration classes which containes the parameters of the models (number of layers, dimensionalities...) and a few utilities to read and write from JSON configuration files. The respective configuration classes are:


* ``BertConfig`` for ``BertModel`` and BERT classes instances.
* ``OpenAIGPTConfig`` for ``OpenAIGPTModel`` and OpenAI GPT classes instances.
* ``GPT2Config`` for ``GPT2Model`` and OpenAI GPT-2 classes instances.
* ``TransfoXLConfig`` for ``TransfoXLModel`` and Transformer-XL classes instances.

These configuration classes contains a few utilities to load and save configurations:


* ``from_dict(cls, json_object)``\ : A class method to construct a configuration from a Python dictionary of parameters. Returns an instance of the configuration class.
* ``from_json_file(cls, json_file)``\ : A class method to construct a configuration from a json file of parameters. Returns an instance of the configuration class.
* ``to_dict()``\ : Serializes an instance to a Python dictionary. Returns a dictionary.
* ``to_json_string()``\ : Serializes an instance to a JSON string. Returns a string.
* ``to_json_file(json_file_path)``\ : Save an instance to a json file.

Models
^^^^^^

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


12. ``TransfoXLModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.TransfoXLModel
    :members:


13. ``TransfoXLLMHeadModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytorch_pretrained_bert.TransfoXLLMHeadModel
    :members:


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


Tokenizers
^^^^^^^^^^

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

``TransfoXLTokenizer``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``TransfoXLTokenizer`` perform word tokenization. This tokenizer can be used for adaptive softmax and has utilities for counting tokens in a corpus to create a vocabulary ordered by toekn frequency (for adaptive softmax). See the adaptive softmax paper (\ `Efficient softmax approximation for GPUs <http://arxiv.org/abs/1609.04309>`_\ ) for more details.

The API is similar to the API of ``BertTokenizer`` (see above).

Please refer to the doc strings and code in `\ ``tokenization_transfo_xl.py`` <./pytorch_pretrained_bert/tokenization_transfo_xl.py>`_ for the details of these additional methods in ``TransfoXLTokenizer``.

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

Optimizers
^^^^^^^^^^

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

``OpenAIAdam``
~~~~~~~~~~~~~~~~~~

``OpenAIAdam`` is similar to ``BertAdam``.
The differences with ``BertAdam`` is that ``OpenAIAdam`` compensate for bias as in the regular Adam optimizer.

``OpenAIAdam`` accepts the same arguments as ``BertAdam``.

Learning Rate Schedules
~~~~~~~~~~~~~~~~~~~~~~~

The ``.optimization`` module also provides additional schedules in the form of schedule objects that inherit from ``_LRSchedule``.
All ``_LRSchedule`` subclasses accept ``warmup`` and ``t_total`` arguments at construction.
When an ``_LRSchedule`` object is passed into ``BertAdam`` or ``OpenAIAdam``\ ,
the ``warmup`` and ``t_total`` arguments on the optimizer are ignored and the ones in the ``_LRSchedule`` object are used.
An overview of the implemented schedules:


* ``ConstantLR``\ : always returns learning rate 1.
* ``WarmupConstantSchedule``\ : Linearly increases learning rate from 0 to 1 over ``warmup`` fraction of training steps.
    Keeps learning rate equal to 1. after warmup.

  .. image:: docs/imgs/warmup_constant_schedule.png
     :target: docs/imgs/warmup_constant_schedule.png
     :alt:

* ``WarmupLinearSchedule``\ : Linearly increases learning rate from 0 to 1 over ``warmup`` fraction of training steps.
    Linearly decreases learning rate from 1. to 0. over remaining ``1 - warmup`` steps.

  .. image:: docs/imgs/warmup_linear_schedule.png
     :target: docs/imgs/warmup_linear_schedule.png
     :alt:

* ``WarmupCosineSchedule``\ : Linearly increases learning rate from 0 to 1 over ``warmup`` fraction of training steps.
   Decreases learning rate from 1. to 0. over remaining ``1 - warmup`` steps following a cosine curve.
   If ``cycles`` (default=0.5) is different from default, learning rate follows cosine function after warmup.

  .. image:: docs/imgs/warmup_cosine_schedule.png
     :target: docs/imgs/warmup_cosine_schedule.png
     :alt:

* ``WarmupCosineWithHardRestartsSchedule``\ : Linearly increases learning rate from 0 to 1 over ``warmup`` fraction of training steps.
    If ``cycles`` (default=1.) is different from default, learning rate follows ``cycles`` times a cosine decaying learning rate (with hard restarts).

  .. image:: docs/imgs/warmup_cosine_hard_restarts_schedule.png
     :target: docs/imgs/warmup_cosine_hard_restarts_schedule.png
     :alt:

* ``WarmupCosineWithWarmupRestartsSchedule``\ : All training progress is divided in ``cycles`` (default=1.) parts of equal length.
    Every part follows a schedule with the first ``warmup`` fraction of the training steps linearly increasing from 0. to 1.,
    followed by a learning rate decreasing from 1. to 0. following a cosine curve.
    Note that the total number of all warmup steps over all cycles together is equal to ``warmup`` * ``cycles``

  .. image:: docs/imgs/warmup_cosine_warm_restarts_schedule.png
     :target: docs/imgs/warmup_cosine_warm_restarts_schedule.png
     :alt: