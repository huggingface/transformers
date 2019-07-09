Usage
================================================

BERT
^^^^

Here is a quick-start example using ``BertTokenizer``\ , ``BertModel`` and ``BertForMaskedLM`` class with Google AI's pre-trained ``Bert base uncased`` model. See the `doc section <#doc>`_ below for all the details on these classes.

First let's prepare a tokenized input with ``BertTokenizer``

.. code-block:: python

   import torch
   from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

   # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
   import logging
   logging.basicConfig(level=logging.INFO)

   # Load pre-trained model tokenizer (vocabulary)
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

   # Tokenized input
   text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
   tokenized_text = tokenizer.tokenize(text)

   # Mask a token that we will try to predict back with `BertForMaskedLM`
   masked_index = 8
   tokenized_text[masked_index] = '[MASK]'
   assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

   # Convert token to vocabulary indices
   indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
   # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
   segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

   # Convert inputs to PyTorch tensors
   tokens_tensor = torch.tensor([indexed_tokens])
   segments_tensors = torch.tensor([segments_ids])

Let's see how to use ``BertModel`` to get hidden states

.. code-block:: python

   # Load pre-trained model (weights)
   model = BertModel.from_pretrained('bert-base-uncased')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor = tokens_tensor.to('cuda')
   segments_tensors = segments_tensors.to('cuda')
   model.to('cuda')

   # Predict hidden states features for each layer
   with torch.no_grad():
       encoded_layers, _ = model(tokens_tensor, segments_tensors)
   # We have a hidden states for each of the 12 layers in model bert-base-uncased
   assert len(encoded_layers) == 12

And how to use ``BertForMaskedLM``

.. code-block:: python

   # Load pre-trained model (weights)
   model = BertForMaskedLM.from_pretrained('bert-base-uncased')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor = tokens_tensor.to('cuda')
   segments_tensors = segments_tensors.to('cuda')
   model.to('cuda')

   # Predict all tokens
   with torch.no_grad():
       predictions = model(tokens_tensor, segments_tensors)

   # confirm we were able to predict 'henson'
   predicted_index = torch.argmax(predictions[0, masked_index]).item()
   predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
   assert predicted_token == 'henson'

OpenAI GPT
^^^^^^^^^^

Here is a quick-start example using ``OpenAIGPTTokenizer``\ , ``OpenAIGPTModel`` and ``OpenAIGPTLMHeadModel`` class with OpenAI's pre-trained  model. See the `doc section <#doc>`_ below for all the details on these classes.

First let's prepare a tokenized input with ``OpenAIGPTTokenizer``

.. code-block:: python

   import torch
   from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

   # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
   import logging
   logging.basicConfig(level=logging.INFO)

   # Load pre-trained model tokenizer (vocabulary)
   tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

   # Tokenized input
   text = "Who was Jim Henson ? Jim Henson was a puppeteer"
   tokenized_text = tokenizer.tokenize(text)

   # Convert token to vocabulary indices
   indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

   # Convert inputs to PyTorch tensors
   tokens_tensor = torch.tensor([indexed_tokens])

Let's see how to use ``OpenAIGPTModel`` to get hidden states

.. code-block:: python

   # Load pre-trained model (weights)
   model = OpenAIGPTModel.from_pretrained('openai-gpt')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor = tokens_tensor.to('cuda')
   model.to('cuda')

   # Predict hidden states features for each layer
   with torch.no_grad():
       hidden_states = model(tokens_tensor)

And how to use ``OpenAIGPTLMHeadModel``

.. code-block:: python

   # Load pre-trained model (weights)
   model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor = tokens_tensor.to('cuda')
   model.to('cuda')

   # Predict all tokens
   with torch.no_grad():
       predictions = model(tokens_tensor)

   # get the predicted last token
   predicted_index = torch.argmax(predictions[0, -1, :]).item()
   predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
   assert predicted_token == '.</w>'

And how to use ``OpenAIGPTDoubleHeadsModel``

.. code-block:: python

   # Load pre-trained model (weights)
   model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
   model.eval()

   #  Prepare tokenized input
   text1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
   text2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
   tokenized_text1 = tokenizer.tokenize(text1)
   tokenized_text2 = tokenizer.tokenize(text2)
   indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
   indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
   tokens_tensor = torch.tensor([[indexed_tokens1, indexed_tokens2]])
   mc_token_ids = torch.LongTensor([[len(tokenized_text1)-1, len(tokenized_text2)-1]])

   # Predict hidden states features for each layer
   with torch.no_grad():
       lm_logits, multiple_choice_logits = model(tokens_tensor, mc_token_ids)

Transformer-XL
^^^^^^^^^^^^^^

Here is a quick-start example using ``TransfoXLTokenizer``\ , ``TransfoXLModel`` and ``TransfoXLModelLMHeadModel`` class with the Transformer-XL model pre-trained on WikiText-103. See the `doc section <#doc>`_ below for all the details on these classes.

First let's prepare a tokenized input with ``TransfoXLTokenizer``

.. code-block:: python

   import torch
   from pytorch_transformers import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

   # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
   import logging
   logging.basicConfig(level=logging.INFO)

   # Load pre-trained model tokenizer (vocabulary from wikitext 103)
   tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

   # Tokenized input
   text_1 = "Who was Jim Henson ?"
   text_2 = "Jim Henson was a puppeteer"
   tokenized_text_1 = tokenizer.tokenize(text_1)
   tokenized_text_2 = tokenizer.tokenize(text_2)

   # Convert token to vocabulary indices
   indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
   indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)

   # Convert inputs to PyTorch tensors
   tokens_tensor_1 = torch.tensor([indexed_tokens_1])
   tokens_tensor_2 = torch.tensor([indexed_tokens_2])

Let's see how to use ``TransfoXLModel`` to get hidden states

.. code-block:: python

   # Load pre-trained model (weights)
   model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor_1 = tokens_tensor_1.to('cuda')
   tokens_tensor_2 = tokens_tensor_2.to('cuda')
   model.to('cuda')

   with torch.no_grad():
       # Predict hidden states features for each layer
       hidden_states_1, mems_1 = model(tokens_tensor_1)
       # We can re-use the memory cells in a subsequent call to attend a longer context
       hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

And how to use ``TransfoXLLMHeadModel``

.. code-block:: python

   # Load pre-trained model (weights)
   model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor_1 = tokens_tensor_1.to('cuda')
   tokens_tensor_2 = tokens_tensor_2.to('cuda')
   model.to('cuda')

   with torch.no_grad():
       # Predict all tokens
       predictions_1, mems_1 = model(tokens_tensor_1)
       # We can re-use the memory cells in a subsequent call to attend a longer context
       predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

   # get the predicted last token
   predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
   predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
   assert predicted_token == 'who'

OpenAI GPT-2
^^^^^^^^^^^^

Here is a quick-start example using ``GPT2Tokenizer``\ , ``GPT2Model`` and ``GPT2LMHeadModel`` class with OpenAI's pre-trained  model. See the `doc section <#doc>`_ below for all the details on these classes.

First let's prepare a tokenized input with ``GPT2Tokenizer``

.. code-block:: python

   import torch
   from pytorch_transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

   # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
   import logging
   logging.basicConfig(level=logging.INFO)

   # Load pre-trained model tokenizer (vocabulary)
   tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

   # Encode some inputs
   text_1 = "Who was Jim Henson ?"
   text_2 = "Jim Henson was a puppeteer"
   indexed_tokens_1 = tokenizer.encode(text_1)
   indexed_tokens_2 = tokenizer.encode(text_2)

   # Convert inputs to PyTorch tensors
   tokens_tensor_1 = torch.tensor([indexed_tokens_1])
   tokens_tensor_2 = torch.tensor([indexed_tokens_2])

Let's see how to use ``GPT2Model`` to get hidden states

.. code-block:: python

   # Load pre-trained model (weights)
   model = GPT2Model.from_pretrained('gpt2')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor_1 = tokens_tensor_1.to('cuda')
   tokens_tensor_2 = tokens_tensor_2.to('cuda')
   model.to('cuda')

   # Predict hidden states features for each layer
   with torch.no_grad():
       hidden_states_1, past = model(tokens_tensor_1)
       # past can be used to reuse precomputed hidden state in a subsequent predictions
       # (see beam-search examples in the run_gpt2.py example).
       hidden_states_2, past = model(tokens_tensor_2, past=past)

And how to use ``GPT2LMHeadModel``

.. code-block:: python

   # Load pre-trained model (weights)
   model = GPT2LMHeadModel.from_pretrained('gpt2')
   model.eval()

   # If you have a GPU, put everything on cuda
   tokens_tensor_1 = tokens_tensor_1.to('cuda')
   tokens_tensor_2 = tokens_tensor_2.to('cuda')
   model.to('cuda')

   # Predict all tokens
   with torch.no_grad():
       predictions_1, past = model(tokens_tensor_1)
       # past can be used to reuse precomputed hidden state in a subsequent predictions
       # (see beam-search examples in the run_gpt2.py example).
       predictions_2, past = model(tokens_tensor_2, past=past)

   # get the predicted last token
   predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
   predicted_token = tokenizer.decode([predicted_index])

And how to use ``GPT2DoubleHeadsModel``

.. code-block:: python

   # Load pre-trained model (weights)
   model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
   model.eval()

   #  Prepare tokenized input
   text1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
   text2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
   tokenized_text1 = tokenizer.tokenize(text1)
   tokenized_text2 = tokenizer.tokenize(text2)
   indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
   indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
   tokens_tensor = torch.tensor([[indexed_tokens1, indexed_tokens2]])
   mc_token_ids = torch.LongTensor([[len(tokenized_text1)-1, len(tokenized_text2)-1]])

   # Predict hidden states features for each layer
   with torch.no_grad():
       lm_logits, multiple_choice_logits, past = model(tokens_tensor, mc_token_ids)
