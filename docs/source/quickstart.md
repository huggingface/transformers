# Quickstart

## Philosophy

Transformers is an opinionated library built for NLP researchers seeking to use/study/extend large-scale transformers models.

The library was designed with two strong goals in mind:

- be as easy and fast to use as possible:

  - we strongly limited the number of user-facing abstractions to learn, in fact there are almost no abstractions, just three standard classes required to use each model: configuration, models and tokenizer,
  - all of these classes can be initialized in a simple and unified way from pretrained instances by using a common `from_pretrained()` instantiation method which will take care of downloading (if needed), caching and loading the related class from a pretrained instance supplied in the library or your own saved instance.
  - as a consequence, this library is NOT a modular toolbox of building blocks for neural nets. If you want to extend/build-upon the library, just use regular Python/PyTorch modules and inherit from the base classes of the library to reuse functionalities like model loading/saving.

- provide state-of-the-art models with performances as close as possible to the original models:

  - we provide at least one example for each architecture which reproduces a result provided by the official authors of said architecture,
  - the code is usually as close to the original code base as possible which means some PyTorch code may be not as *pytorchic* as it could be as a result of being converted TensorFlow code.

A few other goals:

- expose the models' internals as consistently as possible:

  - we give access, using a single API to the full hidden-states and attention weights,
  - tokenizer and base model's API are standardized to easily switch between models.

- incorporate a subjective selection of promising tools for fine-tuning/investigating these models:

  - a simple/consistent way to add new tokens to the vocabulary and embeddings for fine-tuning,
  - simple ways to mask and prune transformer heads.

## Main concepts

The library is build around three type of classes for each models:

- **model classes** which are PyTorch models (`torch.nn.Modules`) of the 8 models architectures currently provided in the library, e.g. `BertModel`
- **configuration classes** which store all the parameters required to build a model, e.g. `BertConfig`. You don't always need to instantiate these your-self, in particular if you are using a pretrained model without any modification, creating the model will automatically take care of instantiating the configuration (which is part of the model)
- **tokenizer classes** which store the vocabulary for each model and provide methods for encoding/decoding strings in list of token embeddings indices to be fed to a model, e.g. `BertTokenizer`

All these classes can be instantiated from pretrained instances and saved locally using two methods:

- `from_pretrained()` let you instantiate a model/configuration/tokenizer from a pretrained version either provided by the library itself (currently 27 models are provided as listed [here](https://huggingface.co/transformers/pretrained_models.html)) or stored locally (or on a server) by the user,
- `save_pretrained()` let you save a model/configuration/tokenizer locally so that it can be reloaded using `from_pretrained()`.

We'll finish this quickstart tour by going through a few simple quick-start examples to see how we can instantiate and use these classes. The rest of the documentation is organized in two parts:

- the **MAIN CLASSES** section details the common functionalities/method/attributes of the three main type of classes (configuration, model, tokenizer) plus some optimization related classes provided as utilities for training,
- the **PACKAGE REFERENCE** section details all the variants of each class for each model architectures and in particular the input/output that you should expect when calling each of them.

## Quick tour: Usage

Here are two examples showcasing a few `Bert` and `GPT2` classes and pre-trained models.

See full API reference for examples for each model class.

### BERT example

Let's start by preparing a tokenized input (a list of token embeddings indices to be fed to Bert) from a text string using `BertTokenizer`

```python
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
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
```

Let's see how we can use `BertModel` to encode our inputs in hidden-states:

```python
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
```

And how to use `BertForMaskedLM` to predict a masked token:

```python
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'
```

### OpenAI GPT-2

Here is a quick-start example using `GPT2Tokenizer` and `GPT2LMHeadModel` class with OpenAI's pre-trained model to predict the next token from a text prompt.

First let's prepare a tokenized input from our text string using `GPT2Tokenizer`

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
text = "Who was Jim Henson ? Jim Henson was a"
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])
```

Let's see how to use `GPT2LMHeadModel` to generate the next token following our text:

```python
# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

# get the predicted next sub-word (in our case, the word 'man')
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'
```

Examples for each model class of each model architecture (Bert, GPT, GPT-2, Transformer-XL, XLNet and XLM) can be found in the [documentation](#documentation).

#### Using the past

GPT-2 as well as some other models (GPT, XLNet, Transfo-XL, CTRL) make use of a `past` or `mems` attribute which can be used to prevent re-computing the key/value pairs when using sequential decoding. It is useful when generating sequences as a big part of the attention mechanism benefits from previous computations.

Here is a fully-working example using the `past` with `GPT2LMHeadModel` and argmax decoding (which should only be used as an example, as argmax decoding introduces a lot of repetition):

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer.encode("The Manhattan bridge")
context = torch.tensor([generated])
past = None

for i in range(100):
    print(i)
    output, past = model(context, past=past)
    token = torch.argmax(output[0, :])

    generated += [token.tolist()]
    context = token.unsqueeze(0)

sequence = tokenizer.decode(generated)

print(sequence)
```

The model only requires a single token as input as all the previous tokens' key/value pairs are contained in the `past`.