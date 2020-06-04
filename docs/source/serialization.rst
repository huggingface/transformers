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
    * ``bert-base-german-cased``: Trained on German data only, 12-layer, 768-hidden, 12-heads, 110M parameters `Performance Evaluation <https://deepset.ai/german-bert>`__
    * ``bert-large-uncased-whole-word-masking``: 24-layer, 1024-hidden, 16-heads, 340M parameters - Trained with Whole Word Masking (mask all of the the tokens corresponding to a word at once)
    * ``bert-large-cased-whole-word-masking``: 24-layer, 1024-hidden, 16-heads, 340M parameters - Trained with Whole Word Masking (mask all of the the tokens corresponding to a word at once)
    * ``bert-large-uncased-whole-word-masking-finetuned-squad``: The ``bert-large-uncased-whole-word-masking`` model finetuned on SQuAD (using the ``run_bert_squad.py`` examples). Results: *exact_match: 86.91579943235573, f1: 93.1532499015869*
    * ``bert-base-german-dbmdz-cased``: Trained on German data only, 12-layer, 768-hidden, 12-heads, 110M parameters `Performance Evaluation <https://github.com/dbmdz/german-bert>`__
    * ``bert-base-german-dbmdz-uncased``: Trained on (uncased) German data only, 12-layer, 768-hidden, 12-heads, 110M parameters `Performance Evaluation <https://github.com/dbmdz/german-bert>`__
    * ``openai-gpt``: OpenAI GPT English model, 12-layer, 768-hidden, 12-heads, 110M parameters
    * ``gpt2``: OpenAI GPT-2 English model, 12-layer, 768-hidden, 12-heads, 117M parameters
    * ``gpt2-medium``: OpenAI GPT-2 English model, 24-layer, 1024-hidden, 16-heads, 345M parameters
    * ``transfo-xl-wt103``: Transformer-XL English model trained on wikitext-103, 18-layer, 1024-hidden, 16-heads, 257M parameters

  *
    a path or url to a pretrained model archive containing:


    * ``bert_config.json`` or ``openai_gpt_config.json`` a configuration file for the model, and
    * ``pytorch_model.bin`` a PyTorch dump of a pre-trained instance of ``BertForPreTraining``\ , ``OpenAIGPTModel``\ , ``TransfoXLModel``\ , ``GPT2LMHeadModel`` (saved with the usual ``torch.save()``\ )

  If ``PRE_TRAINED_MODEL_NAME_OR_PATH`` is a shortcut name, the pre-trained weights will be downloaded from AWS S3 (see the links `here <https://github.com/huggingface/transformers/blob/master/transformers/modeling_bert.py>`__\ ) and stored in a cache folder to avoid future download (the cache folder can be found at ``~/.pytorch_pretrained_bert/``\ ).

*
  ``cache_dir`` can be an optional path to a specific directory to download and cache the pre-trained model weights. This option is useful in particular when you are using distributed training: to avoid concurrent access to the same weights you can set for example ``cache_dir='./pretrained_model_{}'.format(args.local_rank)`` (see the section on distributed training for more information).

* ``from_tf``\ : should we load the weights from a locally saved TensorFlow checkpoint
* ``state_dict``\ : an optional state dictionary (collections.OrderedDict object) to use instead of Google pre-trained models
* ``*inputs``\ , `**kwargs`: additional input for the specific Bert class (ex: num_labels for BertForSequenceClassification)

``Uncased`` means that the text has been lowercased before WordPiece tokenization, e.g., ``John Smith`` becomes ``john smith``. The Uncased model also strips out any accent markers. ``Cased`` means that the true case and accent markers are preserved. Typically, the Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging). For information about the Multilingual and Chinese model, see the `Multilingual README <https://github.com/google-research/bert/blob/master/multilingual.md>`__ or the original TensorFlow repository.

When using an ``uncased model``\ , make sure your tokenizer has ``do_lower_case=True`` (either in its configuration, or passed as an additional parameter).

Examples:

.. code-block:: python

   # BERT
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenize=True)
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section explain how you can save and re-load a fine-tuned model (BERT, GPT, GPT-2 and Transformer-XL).
There are three types of files you need to save to be able to reload a fine-tuned model:


* the model itself which should be saved following PyTorch serialization `best practices <https://pytorch.org/docs/stable/notes/serialization.html#best-practices>`__\ ,
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

   from transformers import WEIGHTS_NAME, CONFIG_NAME

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
   tokenizer.save_pretrained(output_dir)

   # Step 2: Re-load the saved model and vocabulary

   # Example for a Bert model
   model = BertForQuestionAnswering.from_pretrained(output_dir)
   tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
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

