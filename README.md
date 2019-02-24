# PyTorch Pretrained BERT: The Big & Extending Repository of pretrained Transformers

[![CircleCI](https://circleci.com/gh/huggingface/pytorch-pretrained-BERT.svg?style=svg)](https://circleci.com/gh/huggingface/pytorch-pretrained-BERT)

This repository contains op-for-op PyTorch reimplementations, pre-trained models and fine-tuning examples for:

- [Google's BERT model](https://github.com/google-research/bert),
- [OpenAI's GPT model](https://github.com/openai/finetune-transformer-lm),
- [Google/CMU's Transformer-XL model](https://github.com/kimiyoung/transformer-xl), and
- [OpenAI's GPT-2 model](https://blog.openai.com/better-language-models/).

These implementations have been tested on several datasets (see the examples) and should match the performances of the associated TensorFlow implementations (e.g. ~91 F1 on SQuAD for BERT, ~88 F1 on RocStories for OpenAI GPT and ~18.3 perplexity on WikiText 103 for the Transformer-XL). You can find more details in the [Examples](#examples) section below.

Here are some information on these models:

**BERT** was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
This PyTorch implementation of BERT is provided with [Google's pre-trained models](https://github.com/google-research/bert), examples, notebooks and a command-line interface to load any pre-trained TensorFlow checkpoint for BERT is also provided.

**OpenAI GPT** was released together with the paper [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
This PyTorch implementation of OpenAI GPT is an adaptation of the [PyTorch implementation by HuggingFace](https://github.com/huggingface/pytorch-openai-transformer-lm) and is provided with [OpenAI's pre-trained model](https://github.com/openai/finetune-transformer-lm) and a command-line interface that was used to convert the pre-trained NumPy checkpoint in PyTorch.

**Google/CMU's Transformer-XL** was released together with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
This PyTorch implementation of Transformer-XL is an adaptation of the original [PyTorch implementation](https://github.com/kimiyoung/transformer-xl) which has been slightly modified to match the performances of the TensorFlow implementation and allow to re-use the pretrained weights. A command-line interface is provided to convert TensorFlow checkpoints in PyTorch models.

**OpenAI GPT-2** was released together with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
This PyTorch implementation of OpenAI GPT-2 is an adaptation of the [OpenAI's implementation](https://github.com/openai/gpt-2) and is provided with [OpenAI's pre-trained model](https://github.com/openai/gpt-2) and a command-line interface that was used to convert the TensorFlow checkpoint in PyTorch.


## Content

| Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Overview](#overview) | Overview of the package |
| [Usage](#usage) | Quickstart examples |
| [Doc](#doc) |  Detailed documentation |
| [Examples](#examples) | Detailed examples on how to fine-tune Bert |
| [Notebooks](#notebooks) | Introduction on the provided Jupyter Notebooks |
| [TPU](#tpu) | Notes on TPU support and pretraining scripts |
| [Command-line interface](#Command-line-interface) | Convert a TensorFlow checkpoint in a PyTorch dump |

## Installation

This repo was tested on Python 2.7 and 3.5+ (examples are tested only on python 3.5+) and PyTorch 0.4.1/1.0.0

### With pip

PyTorch pretrained bert can be installed by pip as follows:
```bash
pip install pytorch-pretrained-bert
```

If you want to reproduce the original tokenization process of the `OpenAI GPT` paper, you will need to install `ftfy` (limit to version 4.4.3 if you are using Python 2) and `SpaCy` :
```bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

If you don't install `ftfy` and `SpaCy`, the `OpenAI GPT` tokenizer will default to tokenize using BERT's `BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most usage, don't worry).

### From source

Clone the repository and run:
```bash
pip install [--editable] .
```

Here also, if you want to reproduce the original tokenization process of the `OpenAI GPT` model, you will need to install `ftfy` (limit to version 4.4.3 if you are using Python 2) and `SpaCy` :
```bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

Again, if you don't install `ftfy` and `SpaCy`, the `OpenAI GPT` tokenizer will default to tokenize using BERT's `BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most usage).

A series of tests is included in the [tests folder](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/tests) and can be run using `pytest` (install pytest if needed: `pip install pytest`).

You can run the tests with the command:
```bash
python -m pytest -sv tests/
```

## Overview

This package comprises the following classes that can be imported in Python and are detailed in the [Doc](#doc) section of this readme:

- Eight **Bert** PyTorch models (`torch.nn.Module`) with pre-trained weights (in the [`modeling.py`](./pytorch_pretrained_bert/modeling.py) file):
  - [`BertModel`](./pytorch_pretrained_bert/modeling.py#L556) - raw BERT Transformer model (**fully pre-trained**),
  - [`BertForMaskedLM`](./pytorch_pretrained_bert/modeling.py#L710) - BERT Transformer with the pre-trained masked language modeling head on top (**fully pre-trained**),
  - [`BertForNextSentencePrediction`](./pytorch_pretrained_bert/modeling.py#L771) - BERT Transformer with the pre-trained next sentence prediction classifier on top  (**fully pre-trained**),
  - [`BertForPreTraining`](./pytorch_pretrained_bert/modeling.py#L639) - BERT Transformer with masked language modeling head and next sentence prediction classifier on top (**fully pre-trained**),
  - [`BertForSequenceClassification`](./pytorch_pretrained_bert/modeling.py#L833) - BERT Transformer with a sequence classification head on top (BERT Transformer is **pre-trained**, the sequence classification head **is only initialized and has to be trained**),
  - [`BertForMultipleChoice`](./pytorch_pretrained_bert/modeling.py#L899) - BERT Transformer with a multiple choice head on top (used for task like Swag) (BERT Transformer is **pre-trained**, the multiple choice classification head **is only initialized and has to be trained**),
  - [`BertForTokenClassification`](./pytorch_pretrained_bert/modeling.py#L969) - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**, the token classification head **is only initialized and has to be trained**),
  - [`BertForQuestionAnswering`](./pytorch_pretrained_bert/modeling.py#L1034) - BERT Transformer with a token classification head on top (BERT Transformer is **pre-trained**, the token classification head **is only initialized and has to be trained**).

- Three **OpenAI GPT** PyTorch models (`torch.nn.Module`) with pre-trained weights (in the [`modeling_openai.py`](./pytorch_pretrained_bert/modeling_openai.py) file):
  - [`OpenAIGPTModel`](./pytorch_pretrained_bert/modeling_openai.py#L537) - raw OpenAI GPT Transformer model (**fully pre-trained**),
  - [`OpenAIGPTLMHeadModel`](./pytorch_pretrained_bert/modeling_openai.py#L691) - OpenAI GPT Transformer with the tied language modeling head on top (**fully pre-trained**),
  - [`OpenAIGPTDoubleHeadsModel`](./pytorch_pretrained_bert/modeling_openai.py#L752) - OpenAI GPT Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT Transformer is **pre-trained**, the multiple choice classification head **is only initialized and has to be trained**),

- Two **Transformer-XL** PyTorch models (`torch.nn.Module`) with pre-trained weights (in the [`modeling_transfo_xl.py`](./pytorch_pretrained_bert/modeling_transfo_xl.py) file):
  - [`TransfoXLModel`](./pytorch_pretrained_bert/modeling_transfo_xl.py#L974) - Transformer-XL model which outputs the last hidden state and memory cells (**fully pre-trained**),
  - [`TransfoXLLMHeadModel`](./pytorch_pretrained_bert/modeling_transfo_xl.py#L1236) - Transformer-XL with the tied adaptive softmax head on top for language modeling which outputs the logits/loss and memory cells (**fully pre-trained**),

- Three **OpenAI GPT-2** PyTorch models (`torch.nn.Module`) with pre-trained weights (in the [`modeling_gpt2.py`](./pytorch_pretrained_bert/modeling_gpt2.py) file):
  - [`GPT2Model`](./pytorch_pretrained_bert/modeling_gpt2.py#L537) - raw OpenAI GPT-2 Transformer model (**fully pre-trained**),
  - [`GPT2LMHeadModel`](./pytorch_pretrained_bert/modeling_gpt2.py#L691) - OpenAI GPT-2 Transformer with the tied language modeling head on top (**fully pre-trained**),
  - [`GPT2DoubleHeadsModel`](./pytorch_pretrained_bert/modeling_gpt2.py#L752) - OpenAI GPT-2 Transformer with the tied language modeling head and a multiple choice classification head on top (OpenAI GPT-2 Transformer is **pre-trained**, the multiple choice classification head **is only initialized and has to be trained**),

- Tokenizers for **BERT** (using word-piece) (in the [`tokenization.py`](./pytorch_pretrained_bert/tokenization.py) file):
  - `BasicTokenizer` - basic tokenization (punctuation splitting, lower casing, etc.),
  - `WordpieceTokenizer` - WordPiece tokenization,
  - `BertTokenizer` - perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization.

- Tokenizer for **OpenAI GPT** (using Byte-Pair-Encoding) (in the [`tokenization_openai.py`](./pytorch_pretrained_bert/tokenization_openai.py) file):
  - `OpenAIGPTTokenizer` - perform Byte-Pair-Encoding (BPE) tokenization.

- Tokenizer for **Transformer-XL** (word tokens ordered by frequency for adaptive softmax) (in the [`tokenization_transfo_xl.py`](./pytorch_pretrained_bert/tokenization_transfo_xl.py) file):
  - `OpenAIGPTTokenizer` - perform word tokenization and can order words by frequency in a corpus for use in an adaptive softmax.

- Tokenizer for **OpenAI GPT-2** (using byte-level Byte-Pair-Encoding) (in the [`tokenization_gpt2.py`](./pytorch_pretrained_bert/tokenization_gpt2.py) file):
  - `GPT2Tokenizer` - perform byte-level Byte-Pair-Encoding (BPE) tokenization.

- Optimizer for **BERT** (in the [`optimization.py`](./pytorch_pretrained_bert/optimization.py) file):
  - `BertAdam` - Bert version of Adam algorithm with weight decay fix, warmup and linear decay of the learning rate.

- Optimizer for **OpenAI GPT** (in the [`optimization_openai.py`](./pytorch_pretrained_bert/optimization_openai.py) file):
  - `OpenAIGPTAdam` - OpenAI GPT version of Adam algorithm with weight decay fix, warmup and linear decay of the learning rate.

- Configuration classes for BERT, OpenAI GPT and Transformer-XL (in the respective [`modeling.py`](./pytorch_pretrained_bert/modeling.py), [`modeling_openai.py`](./pytorch_pretrained_bert/modeling_openai.py), [`modeling_transfo_xl.py`](./pytorch_pretrained_bert/modeling_transfo_xl.py) files):
  - `BertConfig` - Configuration class to store the configuration of a `BertModel` with utilities to read and write from JSON configuration files.
  - `OpenAIGPTConfig` - Configuration class to store the configuration of a `OpenAIGPTModel` with utilities to read and write from JSON configuration files.
  - `TransfoXLConfig` - Configuration class to store the configuration of a `TransfoXLModel` with utilities to read and write from JSON configuration files.

The repository further comprises:

- Five examples on how to use **BERT** (in the [`examples` folder](./examples)):
  - [`extract_features.py`](./examples/extract_features.py) - Show how to extract hidden states from an instance of `BertModel`,
  - [`run_classifier.py`](./examples/run_classifier.py) - Show how to fine-tune an instance of `BertForSequenceClassification` on GLUE's MRPC task,
  - [`run_squad.py`](./examples/run_squad.py) - Show how to fine-tune an instance of `BertForQuestionAnswering` on SQuAD v1.0 and SQuAD v2.0 tasks.
  - [`run_swag.py`](./examples/run_swag.py) - Show how to fine-tune an instance of `BertForMultipleChoice` on Swag task.
  - [`run_lm_finetuning.py`](./examples/run_lm_finetuning.py) - Show how to fine-tune an instance of `BertForPretraining` on a target text corpus.

- One example on how to use **OpenAI GPT** (in the [`examples` folder](./examples)):
  - [`run_openai_gpt.py`](./examples/run_openai_gpt.py) - Show how to fine-tune an instance of `OpenGPTDoubleHeadsModel` on the RocStories task.

- One example on how to use **Transformer-XL** (in the [`examples` folder](./examples)):
  - [`run_transfo_xl.py`](./examples/run_transfo_xl.py) - Show how to load and evaluate a pre-trained model of `TransfoXLLMHeadModel` on WikiText 103.

- One example on how to use **OpenAI GPT-2** in the unconditional and interactive mode (in the [`examples` folder](./examples)):
  - [`run_gpt2.py`](./examples/run_gpt2.py) - Show how to use OpenAI GPT-2 an instance of `GPT2LMHeadModel` to generate text (same as the original OpenAI GPT-2 examples).

  These examples are detailed in the [Examples](#examples) section of this readme.

- Three notebooks that were used to check that the TensorFlow and PyTorch models behave identically (in the [`notebooks` folder](./notebooks)):
  - [`Comparing-TF-and-PT-models.ipynb`](./notebooks/Comparing-TF-and-PT-models.ipynb) - Compare the hidden states predicted by `BertModel`,
  - [`Comparing-TF-and-PT-models-SQuAD.ipynb`](./notebooks/Comparing-TF-and-PT-models-SQuAD.ipynb) - Compare the spans predicted by  `BertForQuestionAnswering` instances,
  - [`Comparing-TF-and-PT-models-MLM-NSP.ipynb`](./notebooks/Comparing-TF-and-PT-models-MLM-NSP.ipynb) - Compare the predictions of the `BertForPretraining` instances.

  These notebooks are detailed in the [Notebooks](#notebooks) section of this readme.

- A command-line interface to convert TensorFlow checkpoints (BERT, Transformer-XL) or NumPy checkpoint (OpenAI) in a PyTorch save of the associated PyTorch model:

  This CLI is detailed in the [Command-line interface](#Command-line-interface) section of this readme.

## Usage

### BERT

Here is a quick-start example using `BertTokenizer`, `BertModel` and `BertForMaskedLM` class with Google AI's pre-trained `Bert base uncased` model. See the [doc section](#doc) below for all the details on these classes.

First let's prepare a tokenized input with `BertTokenizer`

```python
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

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
```

Let's see how to use `BertModel` to get hidden states

```python
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
```

And how to use `BertForMaskedLM`

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
    predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'
```

### OpenAI GPT

Here is a quick-start example using `OpenAIGPTTokenizer`, `OpenAIGPTModel` and `OpenAIGPTLMHeadModel` class with OpenAI's pre-trained  model. See the [doc section](#doc) below for all the details on these classes.

First let's prepare a tokenized input with `OpenAIGPTTokenizer`

```python
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

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
```

Let's see how to use `OpenAIGPTModel` to get hidden states

```python
# Load pre-trained model (weights)
model = OpenAIGPTModel.from_pretrained('openai-gpt')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

# Predict hidden states features for each layer
with torch.no_grad():
    hidden_states = model(tokens_tensor)
```

And how to use `OpenAIGPTLMHeadModel`

```python
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
```

### Transformer-XL

Here is a quick-start example using `TransfoXLTokenizer`, `TransfoXLModel` and `TransfoXLModelLMHeadModel` class with the Transformer-XL model pre-trained on WikiText-103. See the [doc section](#doc) below for all the details on these classes.

First let's prepare a tokenized input with `TransfoXLTokenizer`

```python
import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

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
```

Let's see how to use `TransfoXLModel` to get hidden states

```python
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
```

And how to use `TransfoXLLMHeadModel`

```python
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
```

### OpenAI GPT-2

Here is a quick-start example using `GPT2Tokenizer`, `GPT2Model` and `GPT2LMHeadModel` class with OpenAI's pre-trained  model. See the [doc section](#doc) below for all the details on these classes.

First let's prepare a tokenized input with `GPT2Tokenizer`

```python
import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

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
```

Let's see how to use `GPT2Model` to get hidden states

```python
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
```

And how to use `GPT2LMHeadModel`

```python
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
```

## Doc

Here is a detailed documentation of the classes in the package and how to use them:

| Sub-section | Description |
|-|-|
| [Loading Google AI's/OpenAI's pre-trained weigths](#loading-google-ai-or-openai-pre-trained-weigths-or-pytorch-dump) | How to load Google AI/OpenAI's pre-trained weight or a PyTorch saved instance |
| [PyTorch models](#PyTorch-models) | API of the BERT, GPT, GPT-2 and Transformer-XL PyTorch model classes |
| [Tokenizers](#tokenizers) | API of the tokenizers class for BERT, GPT, GPT-2 and Transformer-XL|
| [Optimizers](#optimizerss) |  API of the optimizers |

### Loading Google AI or OpenAI pre-trained weigths or PyTorch dump

To load one of Google AI's, OpenAI's pre-trained models or a PyTorch saved model (an instance of `BertForPreTraining` saved with `torch.save()`), the PyTorch model classes and the tokenizer can be instantiated as

```python
model = BERT_CLASS.from_pretrained(PRE_TRAINED_MODEL_NAME_OR_PATH, cache_dir=None)
```

where

- `BERT_CLASS` is either a tokenizer to load the vocabulary (`BertTokenizer` or `OpenAIGPTTokenizer` classes) or one of the eight BERT or three OpenAI GPT PyTorch model classes (to load the pre-trained weights): `BertModel`, `BertForMaskedLM`, `BertForNextSentencePrediction`, `BertForPreTraining`, `BertForSequenceClassification`, `BertForTokenClassification`, `BertForMultipleChoice`, `BertForQuestionAnswering`, `OpenAIGPTModel`, `OpenAIGPTLMHeadModel` or `OpenAIGPTDoubleHeadsModel`, and
- `PRE_TRAINED_MODEL_NAME_OR_PATH` is either:

  - the shortcut name of a Google AI's or OpenAI's pre-trained model selected in the list:

    - `bert-base-uncased`: 12-layer, 768-hidden, 12-heads, 110M parameters
    - `bert-large-uncased`: 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `bert-base-cased`: 12-layer, 768-hidden, 12-heads , 110M parameters
    - `bert-large-cased`: 24-layer, 1024-hidden, 16-heads, 340M parameters
    - `bert-base-multilingual-uncased`: (Orig, not recommended) 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `bert-base-multilingual-cased`: **(New, recommended)** 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `bert-base-chinese`: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `openai-gpt`: OpenAI English model, 12-layer, 768-hidden, 12-heads, 110M parameters
    - `transfo-xl-wt103`: Transformer-XL English model trained on wikitext-103, 18-layer, 1024-hidden, 16-heads, 257M parameters
    - `gpt2`: OpenAI GPT-2 English model, 12-layer, 768-hidden, 12-heads, 117M parameters

  - a path or url to a pretrained model archive containing:

    - `bert_config.json` or `openai_gpt_config.json` a configuration file for the model, and
    - `pytorch_model.bin` a PyTorch dump of a pre-trained instance of `BertForPreTraining`, `OpenAIGPTModel`, `TransfoXLModel`, `GPT2LMHeadModel` (saved with the usual `torch.save()`)

  If `PRE_TRAINED_MODEL_NAME_OR_PATH` is a shortcut name, the pre-trained weights will be downloaded from AWS S3 (see the links [here](pytorch_pretrained_bert/modeling.py)) and stored in a cache folder to avoid future download (the cache folder can be found at `~/.pytorch_pretrained_bert/`).
- `cache_dir` can be an optional path to a specific directory to download and cache the pre-trained model weights. This option is useful in particular when you are using distributed training: to avoid concurrent access to the same weights you can set for example `cache_dir='./pretrained_model_{}'.format(args.local_rank)` (see the section on distributed training for more information).

`Uncased` means that the text has been lowercased before WordPiece tokenization, e.g., `John Smith` becomes `john smith`. The Uncased model also strips out any accent markers. `Cased` means that the true case and accent markers are preserved. Typically, the Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging). For information about the Multilingual and Chinese model, see the [Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md) or the original TensorFlow repository.

**When using an `uncased model`, make sure to pass `--do_lower_case` to the example training scripts (or pass `do_lower_case=True` to FullTokenizer if you're using your own script and loading the tokenizer your-self.).**

Examples:
```python
# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
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

```

### PyTorch models

#### 1. `BertModel`

`BertModel` is the basic BERT Transformer model with a layer of summed token, position and sequence embeddings followed by a series of identical self-attention blocks (12 for BERT-base, 24 for BERT-large).

The inputs and output are **identical to the TensorFlow model inputs and outputs**.

We detail them here. This model takes as *inputs*:
[`modeling.py`](./pytorch_pretrained_bert/modeling.py)
- `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary (see the tokens preprocessing logic in the scripts [`extract_features.py`](./examples/extract_features.py), [`run_classifier.py`](./examples/run_classifier.py) and [`run_squad.py`](./examples/run_squad.py)), and
- `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
- `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. It's a mask to be used if some input sequence lengths are smaller than the max input sequence length of the current batch. It's the mask that we typically use for attention when a batch has varying length sentences.
- `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

This model *outputs* a tuple composed of:

- `encoded_layers`: controled by the value of the `output_encoded_layers` argument:

  - `output_all_encoded_layers=True`: outputs a list of the encoded-hidden-states at the end of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
  - `output_all_encoded_layers=False`: outputs only the encoded-hidden-states corresponding to the last attention block, i.e. a single torch.FloatTensor of size [batch_size, sequence_length, hidden_size],

- `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a classifier pretrained on top of the hidden state associated to the first character of the input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

An example on how to use this class is given in the [`extract_features.py`](./examples/extract_features.py) script which can be used to extract the hidden states of the model for a given input.

#### 2. `BertForPreTraining`

`BertForPreTraining` includes the `BertModel` Transformer followed by the two pre-training heads:

- the masked language modeling head, and
- the next sentence classification head.

*Inputs* comprises the inputs of the [`BertModel`](#-1.-`BertModel`) class plus two optional labels:

- `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size]
- `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size] with indices selected in [0, 1]. 0 => next sentence is the continuation, 1 => next sentence is a random sentence.

*Outputs*:

- if `masked_lm_labels` and `next_sentence_label` are not `None`: Outputs the total_loss which is the sum of the masked language modeling loss and the next sentence classification loss.
- if `masked_lm_labels` or `next_sentence_label` is `None`: Outputs a tuple comprising

  - the masked language modeling logits, and
  - the next sentence classification logits.

An example on how to use this class is given in the [`run_lm_finetuning.py`](./examples/run_lm_finetuning.py) script which can be used to fine-tune the BERT language model on your specific different text corpus. This should improve model performance, if the language style is different from the original BERT training corpus (Wiki + BookCorpus).


#### 3. `BertForMaskedLM`

`BertForMaskedLM` includes the `BertModel` Transformer followed by the (possibly) pre-trained  masked language modeling head.

*Inputs* comprises the inputs of the [`BertModel`](#-1.-`BertModel`) class plus optional label:

- `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size]

*Outputs*:

- if `masked_lm_labels` is not `None`: Outputs the masked language modeling loss.
- if `masked_lm_labels` is `None`: Outputs the masked language modeling logits.

#### 4. `BertForNextSentencePrediction`

`BertForNextSentencePrediction` includes the `BertModel` Transformer followed by the next sentence classification head.

*Inputs* comprises the inputs of the [`BertModel`](#-1.-`BertModel`) class plus an optional label:

- `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size] with indices selected in [0, 1]. 0 => next sentence is the continuation, 1 => next sentence is a random sentence.

*Outputs*:

- if `next_sentence_label` is not `None`: Outputs the next sentence classification loss.
- if `next_sentence_label` is `None`: Outputs the next sentence classification logits.

#### 5. `BertForSequenceClassification`

`BertForSequenceClassification` is a fine-tuning model that includes `BertModel` and a sequence-level (sequence or pair of sequences) classifier on top of the `BertModel`.

The sequence-level classifier is a linear layer that takes as input the last hidden state of the first character in the input sequence (see Figures 3a and 3b in the BERT paper).

An example on how to use this class is given in the [`run_classifier.py`](./examples/run_classifier.py) script which can be used to fine-tune a single sequence (or pair of sequence) classifier using BERT, for example for the MRPC task.

#### 6. `BertForMultipleChoice`

`BertForMultipleChoice` is a fine-tuning model that includes `BertModel` and a linear layer on top of the `BertModel`.

The linear layer outputs a single value for each choice of a multiple choice problem, then all the outputs corresponding to an instance are passed through a softmax to get the model choice.

This implementation is largely inspired by the work of OpenAI in [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) and the answer of Jacob Devlin in the following [issue](https://github.com/google-research/bert/issues/38).

An example on how to use this class is given in the [`run_swag.py`](./examples/run_swag.py) script which can be used to fine-tune a multiple choice classifier using BERT, for example for the Swag task.

#### 7. `BertForTokenClassification`

`BertForTokenClassification` is a fine-tuning model that includes `BertModel` and a token-level classifier on top of the `BertModel`.

The token-level classifier is a linear layer that takes as input the last hidden state of the sequence.

#### 8. `BertForQuestionAnswering`

`BertForQuestionAnswering` is a fine-tuning model that includes `BertModel` with a token-level classifiers on top of the full sequence of last hidden states.

The token-level classifier takes as input the full sequence of the last hidden state and compute several (e.g. two) scores for each tokens that can for example respectively be the score that a given token is a `start_span` and a `end_span` token (see Figures 3c and 3d in the BERT paper).

An example on how to use this class is given in the [`run_squad.py`](./examples/run_squad.py) script which can be used to fine-tune a token classifier using BERT, for example for the SQuAD task.

#### 9. `OpenAIGPTModel`

`OpenAIGPTModel` is the basic OpenAI GPT Transformer model with a layer of summed token and position embeddings followed by a series of 12 identical self-attention blocks.

OpenAI GPT use a single embedding matrix to store the word and special embeddings.
Special tokens embeddings are additional tokens that are not pre-trained: `[SEP]`, `[CLS]`...
Special tokens need to be trained during the fine-tuning if you use them.
The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

The embeddings are ordered as follow in the token embeddings matrice:

```python
    [0,                                                         ----------------------
      ...                                                        -> word embeddings
      config.vocab_size - 1,                                     ______________________
      config.vocab_size,
      ...                                                        -> special embeddings
      config.vocab_size + config.n_special - 1]                  ______________________
```

where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
    `total_tokens_embeddings = config.vocab_size + config.n_special`
You should use the associate indices to index the embeddings.

The inputs and output are **identical to the TensorFlow model inputs and outputs**.

We detail them here. This model takes as *inputs*:
[`modeling_openai.py`](./pytorch_pretrained_bert/modeling_openai.py)
- `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length] were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, total_tokens_embeddings[
- `position_ids`: an optional torch.LongTensor with the same shape as input_ids
    with the position indices (selected in the range [0, config.n_positions - 1[.
- `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
    You can use it to add a third type of embedding to each input token in the sequence
    (the previous two being the word and position embeddings). The input, position and token_type embeddings are summed inside the Transformer before the first self-attention block.

This model *outputs*:
- `hidden_states`: the encoded-hidden-states at the top of the model as a torch.FloatTensor of size [batch_size, sequence_length, hidden_size] (or more generally [d_1, ..., d_n, hidden_size] were d_1 ... d_n are the dimension of input_ids)

#### 10. `OpenAIGPTLMHeadModel`

`OpenAIGPTLMHeadModel` includes the `OpenAIGPTModel` Transformer followed by a language modeling head with weights tied to the input embeddings (no additional parameters).

*Inputs* are the same as the inputs of the [`OpenAIGPTModel`](#-9.-`OpenAIGPTModel`) class plus optional labels:
- `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size].

*Outputs*:
- if `lm_labels` is not `None`:
  Outputs the language modeling loss.
- else:
  Outputs `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, total_tokens_embeddings] (or more generally [d_1, ..., d_n, total_tokens_embeddings] were d_1 ... d_n are the dimension of input_ids)

#### 11. `OpenAIGPTDoubleHeadsModel`

`OpenAIGPTDoubleHeadsModel` includes the `OpenAIGPTModel` Transformer followed by two heads:
- a language modeling head with weights tied to the input embeddings (no additional parameters) and:
- a multiple choice classifier (linear layer that take as input a hidden state in a sequence to compute a score, see details in paper).

*Inputs* are the same as the inputs of the [`OpenAIGPTModel`](#-9.-`OpenAIGPTModel`) class plus a classification mask and two optional labels:
- `multiple_choice_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token whose hidden state should be used as input for the multiple choice classifier (usually the [CLS] token for each choice).
- `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size].
- `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size] with indices selected in [0, ..., num_choices].

*Outputs*:
- if `lm_labels` and `multiple_choice_labels` are not `None`:
  Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
- else Outputs a tuple with:
  - `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, total_tokens_embeddings]
  - `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]

#### 12. `TransfoXLModel`

The Transformer-XL model is described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context".

Transformer XL use a relative positioning with sinusiodal patterns and adaptive softmax inputs which means that:

- you don't need to specify positioning embeddings indices
- the tokens in the vocabulary have to be sorted to decreasing frequency.

This model takes as *inputs*:
[`modeling_transfo_xl.py`](./pytorch_pretrained_bert/modeling_transfo_xl.py)
- `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the token indices selected in the range [0, self.config.n_token[
- `mems`: an optional memory of hidden states from previous forward passes as a list (num layers) of hidden states at the entry of each layer. Each hidden states has shape [self.config.mem_len, bsz, self.config.d_model]. Note that the first two dimensions are transposed in `mems` with regards to `input_ids`.

This model *outputs* a tuple of (last_hidden_state, new_mems)
- `last_hidden_state`: the encoded-hidden-states at the top of the model as a torch.FloatTensor of size [batch_size, sequence_length, self.config.d_model]
- `new_mems`: list (num layers) of updated mem states at the entry of each layer each mem state is a torch.FloatTensor of size [self.config.mem_len, batch_size, self.config.d_model]. Note that the first two dimensions are transposed in `mems` with regards to `input_ids`.

##### Extracting a list of the hidden states at each layer of the Transformer-XL from `last_hidden_state` and `new_mems`:
The `new_mems` contain all the hidden states PLUS the output of the embeddings (`new_mems[0]`). `new_mems[-1]` is the output of the hidden state of the layer below the last layer and `last_hidden_state` is the output of the last layer (i.E. the input of the softmax when we have a language modeling head on top).

There are two differences between the shapes of `new_mems` and `last_hidden_state`: `new_mems` have transposed first dimensions and are longer (of size `self.config.mem_len`). Here is how to extract the full list of hidden states from the model output:

```python
hidden_states, mems = model(tokens_tensor)
seq_length = hidden_states.size(1)
lower_hidden_states = list(t[-seq_length:, ...].transpose(0, 1) for t in mems)
all_hidden_states = lower_hidden_states + [hidden_states]
```

#### 13. `TransfoXLLMHeadModel`

`TransfoXLLMHeadModel` includes the `TransfoXLModel` Transformer followed by an (adaptive) softmax head with weights tied to the input embeddings.

*Inputs* are the same as the inputs of the [`TransfoXLModel`](#-12.-`TransfoXLModel`) class plus optional labels:
- `target`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the target token indices selected in the range [0, self.config.n_token[

*Outputs* a tuple of (last_hidden_state, new_mems)
- `softmax_output`: output of the (adaptive) softmax:
  - if target is None: Negative log likelihood of shape [batch_size, sequence_length]
  - else: log probabilities of tokens, shape [batch_size, sequence_length, n_tokens]
- `new_mems`: list (num layers) of updated mem states at the entry of each layer each mem state is a torch.FloatTensor of size [self.config.mem_len, batch_size, self.config.d_model]. Note that the first two dimensions are transposed in `mems` with regards to `input_ids`.

#### 14. `GPT2Model`

`GPT2Model` is the OpenAI GPT-2 Transformer model with a layer of summed token and position embeddings followed by a series of 12 identical self-attention blocks.

The inputs and output are **identical to the TensorFlow model inputs and outputs**.

We detail them here. This model takes as *inputs*:
[`modeling_gpt2.py`](./pytorch_pretrained_bert/modeling_gpt2.py)
- `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] (or more generally [d_1, ..., d_n, sequence_length] were d_1 ... d_n are arbitrary dimensions) with the word BPE token indices selected in the range [0, vocab_size[
- `position_ids`: an optional torch.LongTensor with the same shape as input_ids
    with the position indices (selected in the range [0, config.n_positions - 1[.
- `token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
    You can use it to add a third type of embedding to each input token in the sequence
    (the previous two being the word and position embeddings). The input, position and token_type embeddings are summed inside the Transformer before the first self-attention block.
- `past`: an optional list of torch.LongTensor that contains pre-computed hidden-states (key and values in the attention blocks) to speed up sequential decoding (this is the `presents` output of the model, cf. below).

This model *outputs*:
- `hidden_states`: the encoded-hidden-states at the top of the model as a torch.FloatTensor of size [batch_size, sequence_length, hidden_size] (or more generally [d_1, ..., d_n, hidden_size] were d_1 ... d_n are the dimension of input_ids)
- `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as a torch.FloatTensors. They can be reused to speed up sequential decoding (see the `run_gpt2.py` example).

#### 15. `GPT2LMHeadModel`

`GPT2LMHeadModel` includes the `GPT2Model` Transformer followed by a language modeling head with weights tied to the input embeddings (no additional parameters).

*Inputs* are the same as the inputs of the [`GPT2Model`](#-14.-`GPT2Model`) class plus optional labels:
- `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size].

*Outputs*:
- if `lm_labels` is not `None`:
  Outputs the language modeling loss.
- else: a tuple of
  - `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, sequence_length, total_tokens_embeddings] (or more generally [d_1, ..., d_n, total_tokens_embeddings] were d_1 ... d_n are the dimension of input_ids)
  - `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as a torch.FloatTensors. They can be reused to speed up sequential decoding (see the `run_gpt2.py` example).

#### 16. `GPT2DoubleHeadsModel`

`GPT2DoubleHeadsModel` includes the `GPT2Model` Transformer followed by two heads:
- a language modeling head with weights tied to the input embeddings (no additional parameters) and:
- a multiple choice classifier (linear layer that take as input a hidden state in a sequence to compute a score, see details in paper).

*Inputs* are the same as the inputs of the [`GPT2Model`](#-14.-`GPT2Model`) class plus a classification mask and two optional labels:
- `multiple_choice_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token whose hidden state should be used as input for the multiple choice classifier (usually the [CLS] token for each choice).
- `lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss is only computed for the labels set in [0, ..., vocab_size].
- `multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size] with indices selected in [0, ..., num_choices].

*Outputs*:
- if `lm_labels` and `multiple_choice_labels` are not `None`:
  Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
- else Outputs a tuple with:
  - `lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, total_tokens_embeddings]
  - `multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]
  - `presents`: a list of pre-computed hidden-states (key and values in each attention blocks) as a torch.FloatTensors. They can be reused to speed up sequential decoding (see the `run_gpt2.py` example).


### Tokenizers:

#### `BertTokenizer`

`BertTokenizer` perform end-to-end tokenization, i.e. basic tokenization followed by WordPiece tokenization.

This class has four arguments:

- `vocab_file`: path to a vocabulary file.
- `do_lower_case`: convert text to lower-case while tokenizing. **Default = True**.
- `max_len`: max length to filter the input of the Transformer. Default to pre-trained value for the model if `None`. **Default = None**
- `never_split`: a list of tokens that should not be splitted during tokenization. **Default = `["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]`**

and three methods:

- `tokenize(text)`: convert a `str` in a list of `str` tokens by (1) performing basic tokenization and (2) WordPiece tokenization.
- `convert_tokens_to_ids(tokens)`: convert a list of `str` tokens in a list of `int` indices in the vocabulary.
- `convert_ids_to_tokens(tokens)`: convert a list of `int` indices in a list of `str` tokens in the vocabulary.

Please refer to the doc strings and code in [`tokenization.py`](./pytorch_pretrained_bert/tokenization.py) for the details of the `BasicTokenizer` and `WordpieceTokenizer` classes. In general it is recommended to use `BertTokenizer` unless you know what you are doing.

#### `OpenAIGPTTokenizer`

`OpenAIGPTTokenizer` perform Byte-Pair-Encoding (BPE) tokenization.

This class has four arguments:

- `vocab_file`: path to a vocabulary file.
- `merges_file`: path to a file containing the BPE merges.
- `max_len`: max length to filter the input of the Transformer. Default to pre-trained value for the model if `None`. **Default = None**
- `special_tokens`: a list of tokens to add to the vocabulary for fine-tuning. If SpaCy is not installed and BERT's `BasicTokenizer` is used as the pre-BPE tokenizer, these tokens are not split. **Default= None**

and five methods:

- `tokenize(text)`: convert a `str` in a list of `str` tokens by (1) performing basic tokenization and (2) WordPiece tokenization.
- `convert_tokens_to_ids(tokens)`: convert a list of `str` tokens in a list of `int` indices in the vocabulary.
- `convert_ids_to_tokens(tokens)`: convert a list of `int` indices in a list of `str` tokens in the vocabulary.
- `set_special_tokens(self, special_tokens)`: update the list of special tokens (see above arguments)
- `decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)`: decode a list of `int` indices in a string and do some post-processing if needed: (i) remove special tokens from the output and (ii) clean up tokenization spaces.

Please refer to the doc strings and code in [`tokenization_openai.py`](./pytorch_pretrained_bert/tokenization_openai.py) for the details of the `OpenAIGPTTokenizer`.

#### `TransfoXLTokenizer`

`TransfoXLTokenizer` perform word tokenization. This tokenizer can be used for adaptive softmax and has utilities for counting tokens in a corpus to create a vocabulary ordered by toekn frequency (for adaptive softmax). See the adaptive softmax paper ([Efficient softmax approximation for GPUs](http://arxiv.org/abs/1609.04309)) for more details.

Please refer to the doc strings and code in [`tokenization_transfo_xl.py`](./pytorch_pretrained_bert/tokenization_transfo_xl.py) for the details of these additional methods in `TransfoXLTokenizer`.

#### `GPT2Tokenizer`

`GPT2Tokenizer` perform byte-level Byte-Pair-Encoding (BPE) tokenization.

This class has three arguments:

- `vocab_file`: path to a vocabulary file.
- `merges_file`: path to a file containing the BPE merges.
- `errors`: How to handle unicode decoding errors. **Default = `replace`**

and two methods:

- `encode(text)`: convert a `str` in a list of `int` tokens by performing byte-level BPE.
- `decode(tokens)`: convert back a list of `int` tokens in a `str`.

Please refer to [`tokenization_gpt2.py`](./pytorch_pretrained_bert/tokenization_gpt2.py) for more details on the `GPT2Tokenizer`.


### Optimizers:

#### `BertAdam`

`BertAdam` is a `torch.optimizer` adapted to be closer to the optimizer used in the TensorFlow implementation of Bert. The differences with PyTorch Adam optimizer are the following:

- BertAdam implements weight decay fix,
- BertAdam doesn't compensate for bias as in the regular Adam optimizer.

The optimizer accepts the following arguments:

- `lr` : learning rate
- `warmup` : portion of `t_total` for the warmup, `-1`  means no warmup. Default : `-1`
- `t_total` : total number of training steps for the learning
    rate schedule, `-1`  means constant learning rate. Default : `-1`
- `schedule` : schedule to use for the warmup (see above). Default : `'warmup_linear'`
- `b1` : Adams b1. Default : `0.9`
- `b2` : Adams b2. Default : `0.999`
- `e` : Adams epsilon. Default : `1e-6`
- `weight_decay:` Weight decay. Default : `0.01`
- `max_grad_norm` : Maximum norm for the gradients (`-1` means no clipping). Default : `1.0`

#### `OpenAIGPTAdam`

`OpenAIGPTAdam` is similar to `BertAdam`.
The differences with `BertAdam` is that `OpenAIGPTAdam` compensate for bias as in the regular Adam optimizer.

`OpenAIGPTAdam` accepts the same arguments as `BertAdam`.

## Examples

| Sub-section | Description |
|-|-|
| [Training large models: introduction, tools and examples](#Training-large-models-introduction,-tools-and-examples) | How to use gradient-accumulation, multi-gpu training, distributed training, optimize on CPU and 16-bits training to train Bert models |
| [Fine-tuning with BERT: running the examples](#Fine-tuning-with-BERT-running-the-examples) | Running the examples in [`./examples`](./examples/): `extract_classif.py`, `run_classifier.py`, `run_squad.py` and `run_lm_finetuning.py` |
| [Fine-tuning BERT-large on GPUs](#Fine-tuning-BERT-large-on-GPUs) | How to fine tune `BERT large`|

### Training large models: introduction, tools and examples

BERT-base and BERT-large are respectively 110M and 340M parameters models and it can be difficult to fine-tune them on a single GPU with the recommended batch size for good performance (in most case a batch size of 32).

To help with fine-tuning these models, we have included several techniques that you can activate in the fine-tuning scripts [`run_classifier.py`](./examples/run_classifier.py) and [`run_squad.py`](./examples/run_squad.py): gradient-accumulation, multi-gpu training, distributed training and 16-bits training . For more details on how to use these techniques you can read [the tips on training large batches in PyTorch](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) that I published earlier this month.

Here is how to use these techniques in our scripts:

- **Gradient Accumulation**: Gradient accumulation can be used by supplying a integer greater than 1 to the `--gradient_accumulation_steps` argument. The batch at each step will be divided by this integer and gradient will be accumulated over `gradient_accumulation_steps` steps.
- **Multi-GPU**: Multi-GPU is automatically activated when several GPUs are detected and the batches are splitted over the GPUs.
- **Distributed training**: Distributed training can be activated by supplying an integer greater or equal to 0 to the `--local_rank` argument (see below).
- **16-bits training**: 16-bits training, also called mixed-precision training, can reduce the memory requirement of your model on the GPU by using half-precision training, basically allowing to double the batch size. If you have a recent GPU (starting from NVIDIA Volta architecture) you should see no decrease in speed. A good introduction to Mixed precision training can be found [here](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) and a full documentation is [here](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html). In our scripts, this option can be activated by setting the `--fp16` flag and you can play with loss scaling using the `--loss_scale` flag (see the previously linked documentation for details on loss scaling). The loss scale can be zero in which case the scale is dynamically adjusted or a positive power of two in which case the scaling is static.

To use 16-bits training and distributed training, you need to install NVIDIA's apex extension [as detailed here](https://github.com/nvidia/apex). You will find more information regarding the internals of `apex` and how to use `apex` in [the doc and the associated repository](https://github.com/nvidia/apex). The results of the tests performed on pytorch-BERT by the NVIDIA team (and my trials at reproducing them) can be consulted in [the relevant PR of the present repository](https://github.com/huggingface/pytorch-pretrained-BERT/pull/116).

Note: To use *Distributed Training*, you will need to run one training script on each of your machines. This can be done for example by running the following command on each server (see [the above mentioned blog post]((https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)) for more details):
```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=$THIS_MACHINE_INDEX --master_addr="192.168.1.1" --master_port=1234 run_classifier.py (--arg1 --arg2 --arg3 and all other arguments of the run_classifier script)
```
Where `$THIS_MACHINE_INDEX` is an sequential index assigned to each of your machine (0, 1, 2...) and the machine with rank 0 has an IP address `192.168.1.1` and an open port `1234`.

### Fine-tuning with BERT: running the examples

We showcase several fine-tuning examples based on (and extended from) [the original implementation](https://github.com/google-research/bert/):

- a *sequence-level classifier* on the MRPC classification corpus,
- a *token-level classifier* on the question answering dataset SQuAD, and
- a *sequence-level multiple-choice classifier* on the SWAG classification corpus.
- a *BERT language model* on another target corpus

#### MRPC

This example code fine-tunes BERT on the Microsoft Research Paraphrase
Corpus (MRPC) corpus and runs in less than 10 minutes on a single K-80 and in 27 seconds (!) on single tesla V100 16GB with apex installed.

Before running this example you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```shell
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
```

Our test ran on a few seeds with [the original implementation hyper-parameters](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks) gave evaluation results between 84% and 88%.

**Fast run with apex and 16 bit precision: fine-tuning on MRPC in 27 seconds!**
First install apex as indicated [here](https://github.com/NVIDIA/apex).
Then run
```shell
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/ \
  --fp16
```

#### SQuAD

This example code fine-tunes BERT on the SQuAD dataset. It runs in 24 min (with BERT-base) or 68 min (with BERT-large) on a single tesla V100 16GB.

The data for SQuAD can be downloaded with the following links and should be saved in a `$SQUAD_DIR` directory.

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```shell
export SQUAD_DIR=/path/to/SQUAD

python run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

Training with the previous hyper-parameters gave us the following results:
```bash
{"f1": 88.52381567990474, "exact_match": 81.22043519394512}
```

#### SWAG

The data for SWAG can be downloaded by cloning the following [repository](https://github.com/rowanz/swagaf)

```shell
export SWAG_DIR=/path/to/SWAG

python run_swag.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_lower_case \
  --do_eval \
  --data_dir $SWAG_DIR/data \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 80 \
  --output_dir /tmp/swag_output/ \
  --gradient_accumulation_steps 4
```

Training with the previous hyper-parameters on a single GPU gave us the following results:
```
eval_accuracy = 0.8062081375587323
eval_loss = 0.5966546792367169
global_step = 13788
loss = 0.06423990014260186
```

#### LM Fine-tuning

The data should be a text file in the same format as [sample_text.txt](./samples/sample_text.txt)  (one sentence per line, docs separated by empty line).
You can download an [exemplary training corpus](https://ext-bert-sample.obs.eu-de.otc.t-systems.com/small_wiki_sentence_corpus.txt) generated from wikipedia articles and splitted into ~500k sentences with spaCy.
Training one epoch on this corpus takes about 1:20h on 4 x NVIDIA Tesla P100 with `train_batch_size=200` and `max_seq_length=128`:


```shell
python run_lm_finetuning.py \
  --bert_model bert-base-uncased \
  --do_lower_case \
  --do_train \
  --train_file ../samples/sample_text.txt \
  --output_dir models \
  --num_train_epochs 5.0 \
  --learning_rate 3e-5 \
  --train_batch_size 32 \
  --max_seq_length 128 \
```

### OpenAI GPT, Transformer-XL and GPT-2: running the examples

We provide three examples of scripts for OpenAI GPT, Transformer-XL and OpenAI GPT-2 based on (and extended from) the respective original implementations:

- fine-tuning OpenAI GPT on the ROCStories dataset
- evaluating Transformer-XL on Wikitext 103
- unconditional and conditional generation from a pre-trained OpenAI GPT-2 model

#### Fine-tuning OpenAI GPT on the RocStories dataset

This example code fine-tunes OpenAI GPT on the RocStories dataset.

Before running this example you should download the
[RocStories dataset](https://github.com/snigdhac/StoryComprehension_EMNLP/tree/master/Dataset/RoCStories) and unpack it to some directory `$ROC_STORIES_DIR`.

```shell
export ROC_STORIES_DIR=/path/to/RocStories

python run_openai_gpt.py \
  --model_name openai-gpt \
  --do_train \
  --do_eval \
  --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
  --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
  --output_dir ../log \
  --train_batch_size 16 \
```

This command runs in about 10 min on a single K-80 an gives an evaluation accuracy of about 86.4% (the authors report a median accuracy with the TensorFlow code of 85.8% and the OpenAI GPT paper reports a best single run accuracy of 86.5%).

#### Evaluating the pre-trained Transformer-XL on the WikiText 103 dataset

This example code evaluate the pre-trained Transformer-XL on the WikiText 103 dataset.
This command will download a pre-processed version of the WikiText 103 dataset in which the vocabulary has been computed.

```shell
python run_transfo_xl.py --work_dir ../log
```

This command runs in about 1 min on a V100 and gives an evaluation perplexity of 18.22 on WikiText-103 (the authors report a perplexity of about 18.3 on this dataset with the TensorFlow code).

#### Unconditional and conditional generation from OpenAI's GPT-2 model

This example code is identical to the original unconditional and conditional generation codes.

Conditional generation:
```shell
python run_gpt2.py
```

Unconditional generation:
```shell
python run_gpt2.py --unconditional
```

The same option as in the original scripts are provided, please refere to the code of the example and the original repository of OpenAI.

## Fine-tuning BERT-large on GPUs

The options we list above allow to fine-tune BERT-large rather easily on GPU(s) instead of the TPU used by the original implementation.

For example, fine-tuning BERT-large on SQuAD can be done on a server with 4 k-80 (these are pretty old now) in 18 hours. Our results are similar to the TensorFlow implementation results (actually slightly higher):
```bash
{"exact_match": 84.56953642384106, "f1": 91.04028647786927}
```
To get these results we used a combination of:
- multi-GPU training (automatically activated on a multi-GPU server),
- 2 steps of gradient accumulation and
- perform the optimization step on CPU to store Adam's averages in RAM.

Here is the full list of hyper-parameters for this run:
```bash
python ./run_squad.py \
  --bert_model bert-large-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_TRAIN \
  --predict_file $SQUAD_EVAL \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 24 \
  --gradient_accumulation_steps 2
```

If you have a recent GPU (starting from NVIDIA Volta series), you should try **16-bit fine-tuning** (FP16).

Here is an example of hyper-parameters for a FP16 run we tried:
```bash
python ./run_squad.py \
  --bert_model bert-large-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_TRAIN \
  --predict_file $SQUAD_EVAL \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR \
  --train_batch_size 24 \
  --fp16 \
  --loss_scale 128
```

The results were similar to the above FP32 results (actually slightly higher):
```bash
{"exact_match": 84.65468306527909, "f1": 91.238669287002}
```

## Notebooks

We include [three Jupyter Notebooks](https://github.com/huggingface/pytorch-pretrained-BERT/tree/master/notebooks) that can be used to check that the predictions of the PyTorch model are identical to the predictions of the original TensorFlow model.

- The first NoteBook ([Comparing-TF-and-PT-models.ipynb](./notebooks/Comparing-TF-and-PT-models.ipynb)) extracts the hidden states of a full sequence on each layers of the TensorFlow and the PyTorch models and computes the standard deviation between them. In the given example, we get a standard deviation of 1.5e-7 to 9e-7 on the various hidden state of the models.

- The second NoteBook ([Comparing-TF-and-PT-models-SQuAD.ipynb](./notebooks/Comparing-TF-and-PT-models-SQuAD.ipynb)) compares the loss computed by the TensorFlow and the PyTorch models for identical initialization of the fine-tuning layer of the `BertForQuestionAnswering` and computes the standard deviation between them. In the given example, we get a standard deviation of 2.5e-7 between the models.

- The third NoteBook ([Comparing-TF-and-PT-models-MLM-NSP.ipynb](./notebooks/Comparing-TF-and-PT-models-MLM-NSP.ipynb)) compares the predictions computed by the TensorFlow and the PyTorch models for masked token language modeling using the pre-trained masked language modeling model.

Please follow the instructions given in the notebooks to run and modify them.

## Command-line interface

A command-line interface is provided to convert a TensorFlow checkpoint in a PyTorch dump of the `BertForPreTraining` class  (for BERT) or NumPy checkpoint in a PyTorch dump of the `OpenAIGPTModel` class  (for OpenAI GPT).

### BERT

You can convert any TensorFlow checkpoint for BERT (in particular [the pre-trained models released by Google](https://github.com/google-research/bert#pre-trained-models)) in a PyTorch save file by using the [`./pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py`](convert_tf_checkpoint_to_pytorch.py) script.

This CLI takes as input a TensorFlow checkpoint (three files starting with `bert_model.ckpt`) and the associated configuration file (`bert_config.json`), and creates a PyTorch model for this configuration, loads the weights from the TensorFlow checkpoint in the PyTorch model and saves the resulting model in a standard PyTorch save file that can be imported using `torch.load()` (see examples in [`extract_features.py`](./examples/extract_features.py), [`run_classifier.py`](./examples/run_classifier.py) and [`run_squad.py`]((./examples/run_squad.py))).

You only need to run this conversion script **once** to get a PyTorch model. You can then disregard the TensorFlow checkpoint (the three files starting with `bert_model.ckpt`) but be sure to keep the configuration file (`bert_config.json`) and the vocabulary file (`vocab.txt`) as these are needed for the PyTorch model too.

To run this specific conversion script you will need to have TensorFlow and PyTorch installed (`pip install tensorflow`). The rest of the repository only requires PyTorch.

Here is an example of the conversion process for a pre-trained `BERT-Base Uncased` model:

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  $BERT_BASE_DIR/bert_model.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
```

You can download Google's pre-trained models for the conversion [here](https://github.com/google-research/bert#pre-trained-models).

### OpenAI GPT

Here is an example of the conversion process for a pre-trained OpenAI GPT model, assuming that your NumPy checkpoint save as the same format than OpenAI pretrained model (see [here](https://github.com/openai/finetune-transformer-lm))

```shell
export OPENAI_GPT_CHECKPOINT_FOLDER_PATH=/path/to/openai/pretrained/numpy/weights

pytorch_pretrained_bert convert_openai_checkpoint \
  $OPENAI_GPT_CHECKPOINT_FOLDER_PATH \
  $PYTORCH_DUMP_OUTPUT \
  [OPENAI_GPT_CONFIG]
```

### Transformer-XL

Here is an example of the conversion process for a pre-trained Transformer-XL model (see [here](https://github.com/kimiyoung/transformer-xl/tree/master/tf#obtain-and-evaluate-pretrained-sota-models))

```shell
export TRANSFO_XL_CHECKPOINT_FOLDER_PATH=/path/to/transfo/xl/checkpoint

pytorch_pretrained_bert convert_transfo_xl_checkpoint \
  $TRANSFO_XL_CHECKPOINT_FOLDER_PATH \
  $PYTORCH_DUMP_OUTPUT \
  [TRANSFO_XL_CONFIG]
```

### GPT-2

Here is an example of the conversion process for a pre-trained OpenAI's GPT-2 model.

```shell
export GPT2_DIR=/path/to/gpt2/checkpoint

pytorch_pretrained_bert convert_gpt2_checkpoint \
  $GPT2_DIR/model.ckpt \
  $PYTORCH_DUMP_OUTPUT \
  [GPT2_CONFIG]
```

## TPU

TPU support and pretraining scripts

TPU are not supported by the current stable release of PyTorch (0.4.1). However, the next version of PyTorch (v1.0) should support training on TPU and is expected to be released soon (see the recent [official announcement](https://cloud.google.com/blog/products/ai-machine-learning/introducing-pytorch-across-google-cloud)).

We will add TPU support when this next release is published.

The original TensorFlow code further comprises two scripts for pre-training BERT: [create_pretraining_data.py](https://github.com/google-research/bert/blob/master/create_pretraining_data.py) and [run_pretraining.py](https://github.com/google-research/bert/blob/master/run_pretraining.py).

Since, pre-training BERT is a particularly expensive operation that basically requires one or several TPUs to be completed in a reasonable amout of time (see details [here](https://github.com/google-research/bert#pre-training-with-bert)) we have decided to wait for the inclusion of TPU support in PyTorch to convert these pre-training scripts.
