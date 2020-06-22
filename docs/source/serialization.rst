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

